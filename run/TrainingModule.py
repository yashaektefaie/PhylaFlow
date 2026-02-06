import random
import torch, torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from deepspeed.ops.adam import FusedAdam
import wandb
import logging
import gc
import torch.distributed
import gc
import torch
import sys
import os
import torch.nn.functional as F
from ete3 import Tree as EteTree

# Ensure the current directory is in sys.path to import 'phyla'
sys.path.append(os.getcwd())
# Import utilities from the provided codebase
from phyla.utils.utils import load_config
from utils.utils import remove_bit, has_polytomy_fast
from phyla.eval.evo_reasoning_eval import (
    Config,
    load_model,
    _encode_sequences_openfold_style,
)

from utils.random_tree import Tree
from utils.bhv_utils import BHVEncoder
from utils.bhv_movie import build_tree_from_splits
from utils.utils import (
pick_group,
find_polytomy_nodes,
number_to_name_newick,
has_polytomy_fast,
resolve_polytomies_random_deterministic,
_pick_knn_pair
)
from utils.metric_utils import (
    kl_divergence_topological_distributions,
    split_bipartition_frequency_correlation,
    compare_likelihood_distributions,
    compare_branch_length_distributions,
    calculate_norm_rf,
)
from data.dataset import PhylaDataModule
from model.model import TreeDenoiserTokenGT
import numpy as np
import logging
from tqdm import tqdm
from utils.utils import compute_merge_metrics

logger = logging.getLogger(__name__)


class TrainingModule(LightningModule):
    def __init__(
        self,
        model: TreeDenoiserTokenGT,
        dataset: PhylaDataModule,
        lr: float = 1e-4,
        record=False,
        epochs: int = 5000,
        lr_scheduler: str = "default",
        num_annealing_steps: int = 10000,
        num_warmup_steps: int = 1000,
        deepspeed: bool = False,
        logger=None,
        max_num_timesteps: int = 20,
        training_sampling_frequency: int = 100,
        num_samples: int = 100,
        dt: float = 0.1,
        # Figure out how to do typing here
        global_splits=None,
        random_trees=None,
        verbose: bool = False,
        phyla_checkpoint_path=None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.record = record
        self.epochs = epochs
        self.warmup_steps = 400
        self.current_step_value = 0
        self.lr_scheduler = lr_scheduler
        self.num_annealing_steps = num_annealing_steps
        self.num_warmup_steps = num_warmup_steps
        self.dataset = dataset
        self.max_num_timesteps = max_num_timesteps
        self.global_splits = global_splits
        self.random_trees = random_trees
        self.verbose = verbose
        self.training_sampling_frequency = training_sampling_frequency
        self.num_samples = num_samples
        self.dt = dt
        self.train_tokenized_trees = None
        self.train_batched_time = None
        self.train_tree = None

        self.automatic_optimization = False
        self.deepspeed = deepspeed
        self.logger_ = logger
        if verbose:
            logging.getLogger("filelock").setLevel(logging.WARNING)
            logging.getLogger("fsspec").setLevel(logging.WARNING)
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self.phyla_checkpoint_path = phyla_checkpoint_path
        self.phyla_model = None
        self.stepper = 1

        phyla_config_path = "configs/sample_eval_config.yaml"

        if self.phyla_checkpoint_path is not None:
            original_argv = sys.argv
            sys.argv = ["script", phyla_config_path]
            try:
                if not os.path.exists(phyla_config_path):
                    logging.warning(
                        f"Phyla configuration file not found at {phyla_config_path}"
                    )

                config = load_config(Config)
                config.trainer.checkpoint_path = self.phyla_checkpoint_path
                config.eval.device = "cuda" if torch.cuda.is_available() else "cpu"
                loaded = load_model(config=config, random_model=False)
                self.phyla_model = loaded["model"]
                self.phyla_model.eval()
                if verbose:
                    logging.info("Phyla model loaded successfully.")
            except Exception as e:
                logging.warning(f"Failed to load Phyla model: {e}")
            finally:
                sys.argv = original_argv

    def compute_phyla_embeddings(self, sequences, names, device="cuda"):
        """
        Generates Phyla embeddings for a batch of sequences.
        """
        if self.phyla_model is None:
            raise ValueError("Phyla model not loaded.")

        # This utility handles tokenization, padding, and CLS token placement
        batch, _ = _encode_sequences_openfold_style(sequences, names)

        # Generate Embeddings
        with torch.no_grad():
            encoded_seqs = batch["encoded_sequences"].to(device)
            sequence_mask = batch["sequence_mask"].to(device)
            cls_positions = batch["cls_positions"].bool().to(device)

            self.phyla_model.to(device)

            # Handle different forward pass signatures depending on model wrapper
            if "TrainingModule" in str(type(self.phyla_model)):
                embeddings = self.phyla_model(
                    encoded_seqs,
                    cls_token_mask=cls_positions,
                    sequence_mask=sequence_mask,
                )
            else:
                embeddings = self.phyla_model(
                    encoded_seqs,
                    sequence_mask,
                    cls_positions,
                )

        return embeddings

    def forward(
        self, batched_tokenized_trees, t, phyla_embeddings, autoregressive=False
    ):
        if not autoregressive:
            velocity, mask = self.model(
                batched_tokenized_trees,
                t,
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
            )
            edge_split_masks = batched_tokenized_trees[-1]
            edge_mask = batched_tokenized_trees[-2]
            return velocity, edge_split_masks, edge_mask
        else:
            all_group_logits = self.model(
                batched_tokenized_trees,
                t,
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
                autoregressive=True,
            )
            return all_group_logits

    def step(self, batch, eval=False, autoregressive=False):
        logs = {}
        if (
            self.phyla_model is not None
            and batch["phyla_embeddings"] is None
            and "ids" in batch
        ):
            phyla_embeddings_list = []
            for i in range(len(batch["ids"])):
                mapping = batch["mappings"][i]
                num_leaf = batch["num_leaves"][i]
                seqs = []
                names = []
                for idx in range(num_leaf):
                    idx_str = str(idx)
                    taxon_name = mapping.get(idx_str)
                    if taxon_name:
                        seq = self.dataset.name_to_seq.get(taxon_name, "")
                        seqs.append(seq)
                        names.append(taxon_name)
                    else:
                        seqs.append("")
                        names.append("unknown")

                embeddings = self.compute_phyla_embeddings(
                    seqs, names, device=self.device
                )
                # Ensure embeddings are (N, D) not (1, N, D)
                if embeddings.dim() == 3 and embeddings.size(0) == 1:
                    embeddings = embeddings.squeeze(0)
                
                phyla_embeddings_list.append(embeddings)

            batch["phyla_embeddings"] = phyla_embeddings_list

        if not autoregressive:
            v_pred, edge_split_masks, edge_mask = self.forward(
                batch["tokenized_trees"],
                batch["batched_time"],
                batch["phyla_embeddings"],
            )
            if self.train_tokenized_trees is None:
                self.train_tokenized_trees = batch["tokenized_trees"]
                self.train_batched_time = batch["batched_time"]
                self.train_tree = batch["original_trees"]
            # else:
            #     if calculate_norm_rf(batch['original_trees'][0], self.train_tree[0]) != 0:
            #         raise Exception("Training tree topology changed during training!")
            #     elif not torch.equal(batch["tokenized_trees"][0], self.train_tokenized_trees[0]):
            #         import pdb; pdb.set_trace()
            #         raise Exception("Training tokenized trees changed during training!")

            velocity_labels = batch["batched_velocity"]
            num_leaves = batch["num_leaves"]
            gathered_velocity_labels = []
            v_pred_indices = []

            for num in range(len(velocity_labels)):
                sub_gathered_velocity_labels = []
                sub_v_pred_indices = []

                num_leave = num_leaves[num]
                real_max_bit = max(m.bit_length() for m in edge_split_masks[num])
                for vel in velocity_labels[num]:
                    if vel.bit_length() == real_max_bit + 1:
                        vel = remove_bit(vel, num_leave - 1)
                    elif vel.bit_length() > real_max_bit + 1:
                        raise Exception(
                            f"Whoa there is a big problem with this split mask {vel} vs real max {real_max_bit}!"
                        )

                    if vel not in edge_split_masks[num]:
                        print(
                            f"This split {vel} from velocity labels is not in edge splits {edge_split_masks[num]}!"
                        )
                        print([i for i in range(vel.bit_length()) if (vel >> i) & 1])
                        raise Exception("Split not found in edge splits")
                    # else:
                    # 	print("WOOO ONE FOUND")
                    sub_gathered_velocity_labels.append(velocity_labels[num][vel])
                    sub_v_pred_indices.append(edge_split_masks[num].index(vel))

                gathered_velocity_labels.append(
                    torch.tensor(sub_gathered_velocity_labels)
                )
                v_pred_indices.append(torch.tensor(sub_v_pred_indices))

            # gathered_velocity_labels = torch.stack(gathered_velocity_labels)
            # v_pred_indices = torch.stack(v_pred_indices)

            # Fix: Flatten tensors to handle variable number of edges per tree
            preds_list = []
            for b_idx in range(len(v_pred_indices)):
                indices = v_pred_indices[b_idx].to(v_pred.device)
                if indices.numel() > 0:
                    preds = v_pred[b_idx].index_select(0, indices)
                    preds_list.append(preds)

            if len(preds_list) > 0:
                v_pred_gathered = torch.cat(preds_list).squeeze(-1)
                gathered_velocity_labels_flat = torch.cat(gathered_velocity_labels).to(
                    v_pred_gathered.device
                )
                loss = ((v_pred_gathered - gathered_velocity_labels_flat) ** 2).mean()
            else:
                loss = torch.tensor(0.0, device=v_pred.device, requires_grad=True)
            # print("Wow congrats")
            logs["loss"] = loss
            logger.info(f"Velocity loss: {loss.item()}")

            if self.record:
                wandb.log({"train/velocity_loss": loss.item()}, step=self.stepper)
            # import pdb

            # pdb.set_trace()
        else:
            all_group_logits = self.forward(
                batch["tokenized_autoregressive_trees"],
                batch["batched_autoregressive_time"],
                batch["phyla_embeddings"],
                autoregressive=True,
            )

            found = {}
            for merge_cluser in batch["batched_autoregressive_labels"]:
                for res_split, components in merge_cluser:
                    found[res_split] = False

            losses = []

            max_logits = []
            total_metrics = []

            chosen_polytomies = []
            polytomy_logits = []
            polytomy_sizes = []  # Track size of each polytomy encountered

            for group in all_group_logits:
                logits = group["logits"]
                labels = batch["batched_autoregressive_labels"]
                splits_in_polytomy = group["splits_represented"]
                
                # Track polytomy size (number of splits in the polytomy)
                polytomy_sizes.append(len(splits_in_polytomy))

                y = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long).to(
                    logits.device
                )

                for labeled_merge_cluster in labels:
                    idxs = None
                    for resulting_split, components in labeled_merge_cluster:
                        res = all([i in splits_in_polytomy for i in components])

                        if res:
                            found[resulting_split] = True
                            idxs = [splits_in_polytomy.index(i) for i in components]

                            for i in idxs:
                                for j in idxs:
                                    if i != j:
                                        y[i, j] = 1.0

                if y.sum() == 0:
                    chosen_polytomies.append(torch.tensor(0.0))
                else:
                    chosen_polytomies.append(torch.tensor(1.0))

                polytomy_logits.append(group["polytomy_pred"])

                if y.sum() > 0:
                    y = y.float()

                    G = logits.size(0)
                    mask = ~torch.eye(
                        G, dtype=torch.bool, device=logits.device
                    )  # off-diagonal only

                    # optionally only use one triangle (avoid double-counting symmetric pairs)
                    tri = torch.triu(mask, diagonal=1)

                    logits_vec = logits[tri]
                    y_vec = y[tri]

                    # ignore any -inf (if any sneak in beyond diagonal)
                    finite = torch.isfinite(logits_vec)
                    logits_vec = logits_vec[finite]
                    y_vec = y_vec[finite]

                    max_logits.append(torch.sigmoid(logits_vec).max().item())
                    # AUC calculation for logging

                    metrics = compute_merge_metrics(
                        logits_vec, y_vec, threshold_logit=0.0, topk=(1, 5, 10)
                    )
                    total_metrics.append(metrics)

                    pos = (y_vec > 0.5).nonzero(as_tuple=False).squeeze(-1)
                    neg = (y_vec < 0.5).nonzero(as_tuple=False).squeeze(-1)
                    # import pdb; pdb.set_trace()
                    k_neg = 512
                    if neg.numel() > k_neg:
                        # take top-k hardest negatives by score
                        neg_scores = logits_vec[neg]
                        topk = torch.topk(neg_scores, k=k_neg, largest=True).indices
                        neg = neg[topk]

                    idx = torch.cat([pos, neg])
                    tau = 1.0
                    s = logits_vec[idx] / tau

                    lse_all = torch.logsumexp(s, dim=0)
                    lse_pos = torch.logsumexp(s[: pos.numel()], dim=0)

                    loss = lse_all - lse_pos

                    # import pdb; pdb.set_trace()
                    # INITIAL LOSS FUNCTION
                    # class imbalance weighting
                    # pos = y_vec.sum().clamp(min=1.0)
                    # neg = (y_vec.numel() - y_vec.sum()).clamp(min=1.0)
                    # pos_weight = (neg / pos).detach()

                    # loss = F.binary_cross_entropy_with_logits(
                    #     logits_vec, y_vec, pos_weight=pos_weight
                    # )

                    losses.append(loss)

            for i in found:
                if not found[i]:
                    import pdb

                    pdb.set_trace()
                    print(
                        "Missing split: ",
                        [j for j in range(int(i).bit_length()) if (int(i) >> j) & 1],
                    )
                    raise Exception(f"Did not find merge for split {i}!")

            L_polytomy_choosing = None

            if len(chosen_polytomies) > 1:
                polytomy_logits_tensor = torch.stack(polytomy_logits).squeeze(1)
                chosen_polytomies_tensor = torch.stack(chosen_polytomies).to(
                    polytomy_logits_tensor.device
                )

                L_polytomy_choosing = F.binary_cross_entropy_with_logits(
                    polytomy_logits_tensor,
                    chosen_polytomies_tensor,
                )

                logger.info(f"Polytomy choosing loss: {L_polytomy_choosing.item()}")
                if self.record:
                    wandb.log(
                        {"train/polytomy_choosing_loss": L_polytomy_choosing.item()},
                        step=self.stepper,
                    )

            L_merging = torch.stack(losses).mean()
            logs["loss"] = L_merging
            logger.info(f"Autoregressive loss: {L_merging.item()}")
            logger.info(f"Max autoregressive logit: {np.mean(max_logits)}")

            aggregated_metrics = {}
            if len(total_metrics) > 0:
                for key in total_metrics[0]:
                    aggregated_metrics[key] = sum(m[key] for m in total_metrics) / len(
                        total_metrics
                    )

                for key in aggregated_metrics:
                    logger.info(f"{key}: {aggregated_metrics[key]}")

            if L_polytomy_choosing is not None:
                logs["loss"] += L_polytomy_choosing

            # Calculate average polytomy size
            avg_polytomy_size = np.mean(polytomy_sizes) if polytomy_sizes else 0.0
            num_polytomies = len(polytomy_sizes)
            logger.info(f"Average polytomy size: {avg_polytomy_size}")

            if self.record:
                # Batch all metrics into a single wandb.log call to avoid step conflicts
                wandb_metrics = {
                    "train/autoregressive_loss": L_merging.item(),
                    "autoregressive_stats/max_autoregressive_logits": np.mean(max_logits),
                    "autoregressive_stats/avg_polytomy_size": avg_polytomy_size,
                    "autoregressive_stats/num_polytomies": num_polytomies,
                }
                wandb_metrics.update(
                    {f"{key}": aggregated_metrics[key] for key in aggregated_metrics}
                )
                wandb.log(wandb_metrics, step=self.stepper)

        return logs

    def sample(
        self,
        newick_starting_trees: list[str],
        phyla_embeddings,
        num_samples=1,
        max_size_polytomy=25,
        mapping=None,
        T=1.0,
        dt_base=0.02,
        eps_len=1e-8,
        hit_tol=1e-10,
        max_events=1000,
        max_steps=20000,
        KNN_TOPM = 32,
        KNN_TAU = 0.05,
        KNN_STOCHASTIC = False,
    ):

        gt = {158475818957369961082530037760: -0.731665585828483, 158475667841642509253883199488: -0.8988413538718826, 10633823966279326983232708282056441856: -0.20368586326845434, 1297036692682702848: -0.3243564495114172, 5575186299632658880234027781612846343782404: -0.9639204068620865, 5575186310017411073122640408634921532260356: -0.3617431909844142, 5575186299632817356052985151573928873820164: -0.2374435317663224, 5575186299632658261264008138922708894220292: -0.9530019126275265, 18446744074246422528: -0.8388591808965112, 65544: -0.5665090723740024, 22300787733826488258843651194474290477006848: -0.9668680615080342, 2475880078570760618517725188: -0.7128061942398058, 2475880078570760549798248452: -0.772267216617224, 11417981562915327060255522087960440916031832064: -0.4042325641560731, 324518558494130005241672719400960: -0.9457872127340289, 664616533193658392910706523548680192: -0.5935464640962718, 85070601871439417691678863969006649344: -0.8163518920011775, 8921691884004065057751092743946531545399955472: -0.8521763419591382, 85070591730234615865843651995381006336: -0.42188566856064574, 41538394830061755369703031248519184: -0.8722759036024446, 41538394830061755369700832225263632: -0.943427003370581, 21267648011789234332364479824969859072: -0.42183774649226297, 10531000427382356270486649808269054939506982384: -0.7628027356450618, 21267647932561071818100142231425908736: -0.13824650750562706, 41538394830061754793240079921840144: -0.8933152510491086, 41538394830061754793240075626872848: -0.5424099000895698, 12131605409268307000784665072685206076409511936: -0.473487549064635, 11417981562915327060255522087960458508218138624: -0.13048199853398915, 21267647932561071818100142222834925568: -0.4277263676151904, 9223372036854775808: -0.371460974, 21267647932558653966460912964485513216: -0.27606764000000006, 51539607552: -0.789158584543765, 53687091200: -0.317142353850169, 170141345719746060945050695293894393856: -0.1265256327378287, 19961783133764987308900089856: -0.9987741288279605, 19961783133476756932748378112: -0.2628487926438454, 170141345719746060954278570930376540160: -0.9571014370828996, 170141345719746060954274067330749169664: -0.597181905893513, 40564819207303340847894502572064: -0.702835855676214, 274877906944: -0.5316655200000001, 41538374868278621028243970633760768: -0.20280910000000002, 8: -0.6586848399999999, 18446744073709551616: -0.89584757, 42535295865117307932921825928971026432: -0.9283643950000001, 174224571863520493293247799005065324265472: -0.5858926, 32768: -0.8984007, 75557863725914323419136: -0.6043512860000001, 89202980794122492567351798982300853608644608: -0.45812724834873847, 12132128094968697343021699179901705138849710080: -0.7752943077886116, 12132128094968697343021699179901704589093896192: -0.15526474870771043, 12132128094968697338069939022760183489497399296: -0.7437469847521373, 181193554738061313024977710965267937236090880: -0.8185402555711767, 12131779645824970297083352527162173358848868352: -0.7684066502166266, 1298074214633706907132624082829312: -0.13632247594001465, 2787593149816327892691964784081045188247552: -0.6313335999999999, 1208925819614629174706176: -0.97317216, 2: -0.5371667999999999, 4611686018427387904: -0.3274074, 10633823966279326983230456482242756608: -0.7173280900000001, 8590983168: -0.2524487730929251, 340282366920938463464671644124450930688: -0.49503663912851253, 10889035741470030830827987437816582766592: -0.93938013, 4722366482869645213696: -0.13801422000000002, 147573952589676412928: -0.7063541000000001, 340282366920938463463374607431768211456: -0.20999499000000002, 64: -0.19947929999999997, 65536: -0.4950193, 348449143727040986586495598010130648530944: -0.8495285450000002, 33451500615458731910139791480672384360398848: -0.752089247054739, 33451500615458730672199752195292109461274624: -0.43837647874383656, 22301128016193409197307115866118414927937536: -0.4936575036071044, 151115727451828646838272: -0.131550556, 1427971984477040136297538520139641290835894624: -0.2499925784297261, 1427971984477040136297538520139641290827506016: -0.9558975449836806, 1125899906842624: -0.3353244, 2596148429267413814265248164610048: -0.35068829970000004, 1427274915959970654148228735669644149578006560: -0.5444269299889422, 41538374868278621028252766726782992: -0.8406278748947806, 8796093022224: -0.8811318992931401, 16: -0.42314170000000006, 36893488147419103232: -0.7664290000000001, 85070591730234615865843651857942052864: -0.660053212, 20769187434140491141771500015583232: -0.2215423043276592, 20769187434140491105742702996619264: -0.1647015821585011, 20769187434140491105742702728183808: -0.14822959293044752, 2722258935367507707706996859454145691648: -0.39281836300000006, 1180591620717411303424: -0.6707895148, 512: -0.8312887399999999, 10633823966279402541096434196379860992: -0.6911298858840041, 664613997892457936451903530142269440: -0.4648114319175766, 11150372599265311570767859136324180752990208: -0.48775007000000004, 4835703278458516698824704: -0.41556431, 2097152: -0.7364710000000001, 1329552514382095629136716866590343168: -0.7096948819956518, 324518597179756232909806309998592: -0.31729109810505934, 696898370530831709732594436453324963192896: -0.3267846636789428, 1427274915959971287973528997358297487606022176: -0.24884675157384795, 4: -0.37704762, 1297036692682719232: -0.3615048844788768, 5070602438691849468944041639936: -0.2688695683901745, 37778931862957228818432: -0.7454440447960716, 87112285931760246646623899502532662132736: -0.2107455, 37778931862957161709568: -0.475919, 16384: -0.4408901, 649037107316853453566312041152512: -0.6217387369999999, 87112285931760246646623899502532662132864: -0.358355172613539, 22969768235052654921127055246504582040609357824: -0.7487449744916324, 22880565254258532428559703447522281187000713216: -0.9486748980724731, 5192296858544272361496235619647488: -0.405831922649313, 3211307308588409732381143431261364056861184000: -0.3901462842337679, 356811923176489970264571492362373784095690752: -0.702716195388327, 5708990770823839524233446109252884202824663040: -0.7288599401511386, 131072: -0.0835494, 302231454903657293676544: -0.8648667400000001, 3212701110480231147117753133498315792256602112: -0.7134704961369555, 8921691881304070671351199242751199995081265152: -0.28782847889949814, 696898287454081973172991196020261297061888: -0.37211505, 2722258935367507707706996859454145691650: -0.7122678160587917, 295147905179352825856: -0.3938135, 680564733841876926926749214863536422912: -0.90725928, 4096: -0.96656028, 87112285931760246646660792990680081236096: -0.24327849190616255, 21778071482940061661655974875633165533184: -0.2688333, 9444732965739290427392: -0.21709334799999996, 2199023255552: -0.528400094, 5070602400912917605986812821504: -0.16108004, 33500768662435011191613705054773636980116340208: -0.9970736264737174, 33500768662435011191613705054773636980116340720: -0.7993039639208753, 128: -0.31548456739999997, 19807040628566084398385987584: -0.5122394, 4503599627370496: -0.09382170000000001, 18014398509481984: -0.218547123, 8589934592: -0.893909, 2305843009213693952: -0.8655166, 5192296858534827628530496329220096: -0.6328248799999999, 140737488355328: -0.15824300000000002, 20769187434139310514121985316880384: -0.9824172999999999, 309485009821345068728975360: -0.9318898825125838, 170141183460469231731687303715884105728: -0.6956975200000001, 73786976294838206464: -0.2634304, 33500768662437607340042972468587902228281016312: -0.16033868696927603, 33500768662435011191613705054773636980116406264: -0.19914292170543108, 32: -0.92875739, 1024: -0.6786740000000001, 2361183241434822606848: -0.845891492, 89202980794122492566142873162686224433938432: -0.4703248939481339, 89202980794122492566142873162651040061849600: -0.9615993304838134, 20599322253708728900222424449024: -0.5947816016441211, 20282409603651671549847174905856: -0.19076486471783796, 87112285931760246646734579966974919442560: -0.768112570858982, 20282409603651670423947268063232: -0.34382355583823504, 89202980794122492566142873090593446023921664: -0.8101786700000001, 38685626227668133590597632: -0.77295635, 16777216: -0.18239849, 12132130817227632710529425776364566440320434178: -0.608841057288808, 10655247261423298577541240952478236672: -0.23006057998971813, 11984799775805394206677957819068579840: -0.15721375947538477, 178405961588244985132285746181186892047843328: -0.9518266, 77371252455336267181195264: -0.24614069200000002, 33554432: -0.876750338, 87197356533631686064426258830943926091904: -1.0006733544373776, 4398046511104: -0.5591561, 8921779081360598689437157170205362489326047376: -0.44015310322655926, 8921691881345609066181260998120903026329784336: -0.5744778740366061, 67108864: -0.8186042699999999, 356811923176489970264571492362373784095686656: -0.7472491, 154742504910672534362390528: -0.46924256000000003, 524288: -0.14976057, 12132130817227632710529406886898564592995401730: -0.2496524105666533, 309485600117155427434627072: -0.3316258096861924, 83076749736557242056487941267529728: -0.6546566371556344, 309490322483638297079840768: -0.6292476882962794, 1361129488283076107562227329949497294848: -0.7740742897783631, 12250663690251526281842553695413718746112: -0.25211205773482903, 43556142965880200694564405087533512294400: -0.44196607556109313, 17592186306560: -0.1634280114481001, 11150372599265321474892636329173694533337088: -0.6056767722632457, 11150372599265311571372322046131495340343296: -0.1367493995095094, 43556142965880123323311949751266331099136: -0.8309685455821875, 12132130817227632710838916103459890755827793922: -0.3286819489546425, 256: -0.7355025, 1361129467683753853853498429727072845824: -0.5540848, 1267650600228229401496703205376: -0.79602339, 633825300114114700748351602688: -0.299743783, 22835963083295358136546656770638941187364356096: -0.6694938576309417, 281474976710656: -0.32151630000000003, 162259276829213363391578010288128: -0.88759204, 2535301200456458802993406410752: -0.6936673600000001, 10141204801825835211973625643008: -0.78936662, 39614081259438011805985669120: -0.9517860424141675, 549755813888: -0.24018540000000002, 20282409603651670423947251286016: -0.5571252999999999, 8192: -0.95794727, 713623846352979940529142984724747568191373312: -0.355079858, 309485009821345068724781056: -0.25884663, 134217728: -0.19717553, 1393801891821414736609702236951735395418112: -0.6279203871408506, 1393801891820147086009474007550238692212736: -0.485396703169406, 8388608: -0.9310123100000001, 44601490397061246283071436545296723011960832: -0.6926529000000001, 536870912: -0.8547623, 2854495385411919762116571938898990272765493248: -0.65608285, 1237940039285380274899124224: -0.9263920899999999, 4194304: -0.2863618, 22300745198530623141535718272648361505980416: -0.4088007, 9671406556917033397649408: -0.60181799, 262144: -0.10429302, 1393796574908163946345982392040522594123776: -0.277364, 1427274915295313556135363039418089731526754304: -0.42476157175087637, 1427274915295313556135363039418089677839663104: -0.737310793889599, 316912650057057350375249543168: -0.6962051657849782, 1073741824: -0.0916866, 44601490398440450136119750134125136099934208: -0.615230327959706, 44601490398359320497705143452429347094790144: -0.4036354465174233, 2475880078570760549798248448: -0.32329020810000003, 27222589353675077077069968594541456916480: -0.7035646769755317, 5708990770823839524233143877797980545530986496: -0.18182857, 590295810358705651712: -0.554357439, 12132130817227632710529425776364566440320565250: -0.6404433106913857, 1427275086101317007729261358193786550891839520: -0.5483040745247466, 1427275086101317007719589951636869517494190112: -0.14742938559435922, 1427275086101317007719589951636868417982562336: -0.38830548251977737, 2417851639229258349412352: -0.6845067, 1048576: -0.5319881000000001, 5575186299632655785383929568162090376495104: -0.3870277, 696898375723128568276866797949560582840640: -0.6298979055769853, 8796093022208: -0.54615, 9007199254740992: -0.5657858, 144115188075855872: -0.4783032, 1099511627776: -0.15149364, 1329227995784915872903807060280344576: -0.567620313, 576460752303423488: -0.6845397999999999, 2048: -0.95081954, 10384593717069655257060992658440192: -0.49213039000000003, 12132130817227632710529425776364496071576256514: -0.8907233515522223, 5444517870735015415413993718908291383296: -0.5986911520000001, 43556142965880123323311949751266331066368: -0.77128803, 2147483648: -0.18069213, 11417981541647679048466287755595961091061972992: -0.9390782999999999, 4951760157141521099596496896: -0.24220397999999999, 18889465931478580854784: -0.9797005500000001, 12250165229753106938390214767766080061440: -0.4729157109932202, 39614081257132168796771975168: -0.27499411999999995, 12132130817227632710529425780976252458747953154: -0.9095502259197155, 17179869184: -0.46371051, 5316911983139663491615228241121378304: -0.40179855000000003, 288230376151711744: -0.5714318799999999, 36028797018963968: -0.3921306, 34359738368: -0.0639124, 696898287454081973175352379965383695663168: -0.5930554063873374, 696898287454081973175352379261696253886528: -0.13810612990336243, 2361183241434956824640: -0.5926955985703701, 2361183241434822606912: -0.27712471422144497, 79228162514264337593543950336: -0.92442973, 12250331383252580052874327743648648659968: -0.6977918703207109, 12250331383252580052874327743648615105536: -0.15084700393496248, 12250165229753106938390214767766080062464: -0.5330723855527605, 1298074214633706907132624082305024: -0.98062812, 562949953421312: -0.6878934999999999, 10655242190820859885691772008436596736: -0.5252614535752409, 10654593153713543032238205696395444224: -0.5203146689389152, 5316911983139663491615509716098088960: -0.8332862440965919, 332306998946228968225951765070086144: -0.139575868, 2251799813685248: -0.93207731, 2658455991569831745825628519070171136: -0.5627937981158709, 166153499473114484112975882535043072: -0.26980696000000004, 83076749736557242056487941267521536: -0.90755406, 1152921504606846976: -0.59735567, 72057594037927936: -0.6106075100000001, 664613997892457936451903530140172288: -0.49048681, 2658455991569831745807614120560689152: -0.8239110900000001, 158456325028528675187087900672: -0.8376991119999999, 68719476736: -0.7228846600000001, 17592186044416: -0.15298233, 5192296858544272361496235619647744: -0.17328533605422589, 12131779633840170521277958320484215539780288512: -0.6933046330050417, 12131779633840170521277958320484211141733777408: -0.1537968872579432, 40564819207303340847894502572032: -0.6992508000000002, 22835963083295358096932575511200929381378686976: -0.1962921374209446, 4294967296: -0.44879330000000006, 22835963083295358096932575511191922182123945984: -0.29888729999999997, 9903520314283042199192993792: -0.44313440000000004, 1427274915295354120954570342758937626029326368: -0.35556344696422204, 35184372088832: -0.80611311, 81129638414606681695789005144064: -0.8535050000000001, 268435456: -0.8734498, 618970019642690137449562112: -0.42264314999999997, 1427247692705959881058285969449495136382746624: -0.6877839499999999, 19342813113834066795298816: -0.9763735802000001, 12250663690251526281842553695413718748160: -0.5875888014578358, 45632899479665240050881888572047792984108810234: -0.7907101120053491, 22880564573693798586682776520773066323464290304: -0.19299567942349613, 70368744177664: -0.1368941, 696898375723128568277161945854739935666496: -0.6592243905122367, 137438953472: -0.6771687, 316912650057057350374175801344: -0.87581393, 604462909807314587353088: -0.137097543, 1427274915959971287973528849784344897929609248: -0.9716313969391362, 9103028442905316134189111288129413648671087760: -0.10617014865176422, 9102984886762350253988416723724326115158793360: -0.7267591724511383, 9102984886762350253988416723724325840280886416: -0.2794603883653689, 8921791332024288940963439012759057903044795536: -0.21991407944129673, 324518553658426726783156020576256: -0.33176211, 45666350980280698782792028363528465368469209082: -0.7310239660354657, 45671926166590716193865151003937100290001469438: -0.17020763288696578, 703687441776640: -0.6406324161906625}

        self.model.eval()
        max_logits = []

        if (
            phyla_embeddings is None
            and self.phyla_model is not None
            and self.dataset is not None
        ):
            # Calculate embeddings on the fly
            t_temp = Tree(newick_starting_trees[0])
            sorted_names = [t_temp.id_to_name[i] for i in range(t_temp.n_leaves)]

            # Filter out ROOT_DUMMY as it has no sequence
            valid_names = [n for n in sorted_names if n != "ROOT_DUMMY"]
            sorted_seqs = [self.dataset.name_to_seq[name] for name in valid_names]

            raw_emb = self.compute_phyla_embeddings(
                sorted_seqs, valid_names, device=self.device
            )
            # raw_emb is (1, N, D). We want (B, N, D).
            if raw_emb.size(0) == 1:
                phyla_embeddings = raw_emb.expand(len(newick_starting_trees), -1, -1)
            else:
                phyla_embeddings = raw_emb.expand(len(newick_starting_trees), -1, -1)

        # SPEED UP SAMPLING
        # 1) init: parse tree -> {mask: length}
        trees = []
        num_leaves = []
        mapping = []
        # Precompute cache for initial trees
        # Since topology changes in the loop, we will update this cache dynamically
        # Initialize tokenized structure cache
        current_newicks = list(newick_starting_trees)
        token_cache = self.model.tokenizer.create_batched_cache(current_newicks)
        #tokenized = self.dataset.tree_tokenizer(current_newicks[0])
        # new_tokenized = ()
        # for i in tokenized:
        #     if torch.is_tensor(i):
        #         new_tokenized += (i.to(self.device),)
        #     else:
        #         new_tokenized += (i,)


        for nw in newick_starting_trees:
            t = Tree(nw)
            enc = BHVEncoder()
            masks, lens = enc.return_BHV_encoding(t)
            # Initial trees have no polytomies and all lengths should be greater than 0, so any 0 edges need to be removed
            trees.append({m: float(l) for m, l in zip(masks, lens) if l is not None})
            num_leaves.append(t.n_leaves)
            mapping.append(t.id_to_name)

        t = 0.0
        n_events = 0
        n_steps = 0
        num_topology_changes = 0
        polytomy_sizes = []  # Track sizes of polytomies encountered during sampling

        while t < T and n_steps < max_steps and n_events < max_events:
            n_steps += 1

            # --- encode/tokenize current trees for the model ---

            # Use CACHED tokenizer
            tokenized = self.model.tokenizer.forward_batched(token_cache, trees)
            #import pdb; pdb.set_trace()

            # if calculate_norm_rf(current_newicks[0], self.train_tree[0]) != 0:
            #     raise Exception("Current tree does not match training tree topology!")
            # #import pdb; pdb.set_trace()
            # if tokenized[0].shape[1] != self.train_tokenized_trees[0].shape[1]:
            #     raise Exception("Tokenized tree length mismatch!")
            # elif (new_tokenized[0] == self.train_tokenized_trees[0]).all().item() is False:
            #     raise Exception("Tokenized trees do not match!")
            
 
            with torch.no_grad():
                velocity, edge_splits, edge_split_mask = self.forward(
                    tokenized, t, phyla_embeddings
                )

            # ---- FIRST PASS: compute per-tree dt_hit, cache per-tree arrays ----

            dt_hit_list = []
            cache = []
            for b_idx, (td, v, n_leaves, mapp) in enumerate(
                zip(trees, velocity, num_leaves, mapping)
            ):
                model_masks = edge_splits[b_idx]
                mask_idx = {mask: i for i, mask in enumerate(model_masks)}
                V = v.squeeze(1).detach().cpu().numpy()

                L = []
                V_val = []
                masks = []
                pred_velocity_dict = {}
                gt_vel_diff = []

                for m in td:
                    if m not in model_masks:
                        # WE GOTTA FIX THIS LOL
                        print(f"Whoa there is a split missing in velocity masks! {m} or {[i for i in range(m.bit_length()) if (m >> i) & 1]}")
                    else:
                        L.append(td[m])

                        #We should not be making moves based on leafs! If leaf, velocity is 0
                        if m.bit_count() == 1:
                            V_val.append(0.0)
                        else:
                            V_val.append(V[mask_idx[m]])

                        masks.append(m)
                        pred_velocity_dict[m] = V[mask_idx[m]]
                
                # for item in pred_velocity_dict:
                #     if item in gt:
                #         gt_vel_diff.append(pred_velocity_dict[item] - gt[item])  # Ground truth velocity is 0 for all edges
                #     else:
                #         print(f"Missing ground truth for edge {item} split {[i for i in range(item.bit_length()) if (item >> i) & 1]}")
                # mse = np.mean(np.square(gt_vel_diff))
                # import pdb; pdb.set_trace()

                V = np.array(V_val, dtype=np.float64)
                L = np.array(L, dtype=np.float64)

                if len(V) != len(L):
                    raise Exception("I assume these two things are equal length!")

                if (L < 0).any():
                    raise Exception("There are negative lengths that is not possible!")

                eps_dt = max(1e-4, 1e-3 * float(np.median(L)) if len(L) > 0 else 1e-4)

                # --- compute dt_hit ---
                neg = (V < 0) & (L > eps_dt)
                if np.any(neg):
                    dt_candidates = L[neg] / -V[neg]
                    dt_hit = float(np.min(dt_candidates))
                else:
                    dt_hit = float("inf")
                
                cache.append((td, L, V, n_leaves, mapp, dt_hit, dt_candidates, masks, neg))
                dt_hit_list.append(dt_hit)

            # ---- GLOBAL dt across the batch ----
            dt_hit_global = min(dt_hit_list) if len(dt_hit_list) else float("inf")
            # Experimenting here, dt_hit_global is not a good metric we just jump, jump, jump, so why not use dt_base
            dt = min(dt_base, dt_hit_global, T - t)
            
            if dt < 2e-4:
                dt = 2e-4

            # dt = min(dt_base, T - t)
            if dt <= 0:
                dt = min(dt_base, T - t)

            #dt = min(dt_base, T - t)

            # ---- SECOND PASS: advance everyone with the SAME dt ----
            new_trees = []

            # Since update of token_cache happens per tree potentially, we need to defer it or track which ones changed.
            # However, batch indices align with zip(trees...), so we can update token_cache[i] if needed.

            for b_idx, (td, L, V, n_leaves, mapp, dt_hit, dt_candidates, masks, neg) in enumerate(cache):
                model_masks = edge_splits[b_idx]
                # --- advance ---
                L_new = L + dt * V
                # import pdb; pdb.set_trace()
                
                # treat as boundary if we stepped past the first hit time for THIS tree
                # (float equality with hit_tol=1e-10 is too strict)
                hit_boundary = (np.isfinite(dt_hit) and dt >= dt_hit) or (L_new <= eps_len).any()

                if hit_boundary and np.isfinite(dt_hit):
                    # Batch near-simultaneous hits (numerical simultaneity)
                    eta = 0.01  # relative tie threshold (1%)
                    eps_abs = 0.1 * dt_hit  # absolute floor, scaled to event time (often ~1e-3 here)
                    idx_neg = np.where(neg)[0]

                    dt_min = dt_candidates.min()
                    eta = 0.02          # relative window
                    c = 2.0             # absolute window in *steps*
                    eps_abs = c * dt_base

                    batch = (dt_candidates <= dt_min * (1 + eta)) | (dt_candidates <= dt_min + eps_abs)
                    L_new[idx_neg[batch]] = 0.0
                
                L_new[L_new < eps_len] = 0.0

                # update dict
                td2 = {m: float(l) for m, l in zip(masks, L_new) if l > eps_len}

                # We only need to rebuild Newick/Graph if we hit a boundary (topology changed)
                if hit_boundary:
                    logger.info(f"Tree hit a boundary, applying topology changes.")
                    num_merges = 0
                    topology_changed = True
                    polytomy_nodes = True

                    while polytomy_nodes:
                        original_td2 = td2.copy()

                        _, td2_newick= build_tree_from_splits(
                            list(original_td2.keys()),
                            original_td2,
                            n_leaves=n_leaves,
                            root_leaf=n_leaves - 1,
                            mapping=mapp,
                        )

                        # For autoregressive step, we just use standard tokenizer for now as it's rare event
                        tokenized_trees = self.model.tokenizer([td2_newick])
                        # import pdb; pdb.set_trace()
                        
                        polytomy_nodes = has_polytomy_fast(
                            td2_newick, unrooted_ok=False
                        )

                        if not polytomy_nodes:
                            logger.info("No polytomies remain after merges exit out")
                            break

                        with torch.no_grad():
                            logit_outputs = self.forward(
                                tokenized_trees,
                                torch.tensor([num_merges / 63], device=self.device),
                                phyla_embeddings,
                                autoregressive=True,
                            )
                        
                        if len(logit_outputs) == 1:
                            candidates = [logit_outputs[0]]
                        else:
                            scores = torch.stack([o["polytomy_pred"].squeeze() for o in logit_outputs])
                            order = torch.argsort(scores, descending=True)
                            candidates = [logit_outputs[int(order[0])]]

                        top_change = False
                        for output in candidates:
                            # Track polytomy size
                            polytomy_sizes.append(len(output["splits_represented"]))
                            
                            # if torch.sigmoid(output["polytomy_pred"]).item() > 0.5 or len(logit_outputs) == 1:
                            #else:
                            logits = output["logits"]
                            G = logits.size(0)
                            mask = ~torch.eye(
                                G, dtype=torch.bool, device=logits.device
                            )  # off-diagonal only

                            # optionally only use one triangle (avoid double-counting symmetric pairs)
                            tri = torch.triu(mask, diagonal=1)
                            ii, jj = torch.triu_indices(G, G, offset=1, device=logits.device)

                            logits_vec = logits[tri]
                            finite = torch.isfinite(logits_vec)
                            if finite.sum() == 0:
                                res = None
                            else:
                                logits_vec_f = logits_vec[finite]
                                ii_f, jj_f = ii[finite], jj[finite]
                                k = torch.argmax(logits_vec_f)
                                i, j = ii_f[k].item(), jj_f[k].item()
                                res = [i, j]
                                
                            if res is None:
                                logger.info("No merges found!")
                            else:
                                logger.info(f"Merges found: {res}")
                                # import pdb; pdb.set_trace()
                                split_masks = [
                                    output["splits_represented"][idx]
                                    for idx in res
                                ]
                                new_split = 0
                                for sm in split_masks:
                                    new_split |= sm
                                
                                to_print = [i for i in range(new_split.bit_length()) if (new_split >> i) & 1]
                                logger.info(f"Merging splits {split_masks[0]}, {split_masks[1]} to create this split {new_split}: {to_print}")


                                if new_split in td2:
                                    import pdb; pdb.set_trace()
                                    logger.info("Whoa already in there!")
                                    raise Exception("Not possible to merge into a split that already exists...")
                                else:
                                    # New length is average of merged splits
                                    curr_lens = list(td2.values())
                                    if len(curr_lens) > 0:
                                        td2[new_split] = float(np.percentile(curr_lens, 10))
                                    else:
                                        td2[new_split] = 1e-3
                                top_change = True
                                logger.info("Merge performed time to break out")
                                num_merges += 1
                                n_events += 1
                                num_topology_changes += 1
                                break
                        
                        # if not top_change:
                        #     logger.info("No more merges possible, pick a random polytomy and do a KNN merge")
                        #     output = random.choice(logit_outputs)
                        #     split_embeddings = output['group_embeddings']
                        #     group_represented = output['splits_represented']

                        #     if len(group_represented) != split_embeddings.size(0):
                        #         raise Exception("Whoa size mismatch between groups and split embeddings")
                            
                        #     i, j = _pick_knn_pair(split_embeddings, topM=KNN_TOPM, tau=KNN_TAU, stochastic=KNN_STOCHASTIC)

                        #     sm_i, sm_j = group_represented[i], group_represented[j]
                        #     new_split = int(sm_i) | int(sm_j)

                        #     if new_split not in td2:
                        #         # td2[new_split] = 1e-3  # tiny length
                        #         curr_lens = list(td2.values())
                        #         if len(curr_lens) > 0:
                        #             td2[new_split] = float(np.median(curr_lens))
                        #         else:
                        #             td2[new_split] = 1e-3
                        #     else:
                        #         # import pdb; pdb.set_trace()
                        #         raise Exception("Not possible to merge into a split that already exists...")

                        #     top_change = True
                        #     num_merges += 1
                        #     n_events += 1
                        #     num_topology_changes += 1



                    _, td2_newick_final = build_tree_from_splits(
                        list(td2.keys()),
                        td2,
                        n_leaves=n_leaves,
                        root_leaf=n_leaves - 1,
                        mapping=mapp,
                    )
                    # Update the cache for this batch index
                    new_item = self.model.tokenizer.compute_structural_cache(
                        [td2_newick_final]
                    )[0]
                    # print(b_idx, "Topology changed, updating token cache.")
                    # import pdb;pdb.set_trace()
                    token_cache.update(b_idx, new_item)

                new_trees.append(td2)

            trees = new_trees
            t += dt

            if n_steps % 100 == 0:
                print(f"Step {n_steps}: dt={dt:.2e}, t={t:.2f}/{T}")

        # print(f"Sampling finished in {n_steps} steps. Total events: {n_events}")
        avg_polytomy_size = np.mean(polytomy_sizes) if polytomy_sizes else 0.0
        # if num_topology_changes > 0:
        #     import pdb; pdb.set_trace()
    
        logger.info(f"Sampling finished in {n_steps} steps. Total events: {n_events}, topology changes: {num_topology_changes}, average polytomy size: {avg_polytomy_size:.2f}")
        return [
            build_tree_from_splits(
                list(td.keys()),
                td,
                n_leaves=n_leaves,
                root_leaf=n_leaves - 1,
                mapping=mapp,
            )[1]
            for td, n_leaves, mapp in zip(trees, num_leaves, mapping)
        ], num_topology_changes, sum(max_logits) / len(max_logits) if len(max_logits) > 0 else 0.0, avg_polytomy_size, len(polytomy_sizes)

    def sample_compare(self, batch, train=True, num_samples=1, dt=0.02, save=True):
        nexus_filepaths = batch["nexus_filepaths"]
        tree_paths = batch["tree_paths"]
        ids = batch["ids"]

        if len(set(nexus_filepaths)) != 1 or len(set(ids)) != 1:
            raise Exception(
                "Each batch should correspond to one ID, not multiple different IDs, logic is inconsitent somewhere"
            )

        nexus_filepath = batch["nexus_filepaths"][0]
        id = batch["ids"][0]
        mapping = batch["mappings"][0]
        seq_ordering_map = batch["sequence_ordering_maps"][0]

        if train:
            real_trees = self.dataset.dataset_train.return_posterior_trees(id)
            num_leaves = self.dataset.dataset_train.return_number_leaves(id)
        else:
            real_trees = self.dataset.dataset_val.return_posterior_trees(id)
            num_leaves = self.dataset.dataset_val.return_number_leaves(id)

        if len(real_trees) > num_samples:
            pot_real_trees = random.sample(real_trees, num_samples)
        else:
            pot_real_trees = real_trees

        sanity_check = self.dataset.dataset_train.sanity_check if train else self.dataset.dataset_val.sanity_check
        random_sanity_check = self.dataset.dataset_train.random_sanity_check if train else self.dataset.dataset_val.random_sanity_check
        
        real_trees = []
        for i in pot_real_trees:
            t_real = EteTree(i, format=1)
            for leaf in t_real.get_leaves():
                name = leaf.name
                # direct match (most likely)
                if name in seq_ordering_map:
                    leaf.name = seq_ordering_map[name]
                else:
                    import pdb; pdb.set_trace()
                    raise Exception("Leaf name in real tree not found in original names map!")
            real_trees.append(t_real.write(format=1))


        for i in real_trees:
            if has_polytomy_fast(i):
                raise Exception(
                    "Whoa there is a polytomy in the real trees, need to resolve first!"
                )

        sampled_trees = []
        num_topology_changes = []
        avg_max_logits = []
        num_polytomies = 0
        starting_trees_nw = []

        avg_polytomy_sizes = []
        num_polytomies_resolved = []

        for _ in tqdm(range(num_samples)):
            # rt = Tree(num_leaves=num_leaves, random=True)
            # starting_tree = str(rt)
            if train:
                starting_tree = self.dataset.dataset_train.sample_random_tree(
                    real_trees[0]
                )
            else:
                starting_tree = self.dataset.dataset_val.sample_random_tree(
                    real_trees[0]
                )
            
            #Now remap the random tree to make the indices match up with the real tree
            t_random = EteTree(starting_tree, format=1)

            for leaf in t_random.get_leaves():
                name = leaf.name
                #Random sanity check to offset by 1
                if not sanity_check:
                    name = str(int(leaf.name)+1)
                # direct match (most likely)
                if name in seq_ordering_map:
                    leaf.name = seq_ordering_map[name]
                else:
                    import pdb; pdb.set_trace()
                    raise Exception("Leaf name in random tree not found in original names map!")
            starting_tree = t_random.write(format=1)


            #### DEBUG CHANGE LATER MADE ONE TIMEPOINT ####
            timepoint = random.uniform(0, 1)

            starting_trees_nw.append(starting_tree)
            sampled_tree, n_topology_changes, avg_max_logit, avg_polytomy_size, n_polytomies_resolved = self.sample(
                [starting_tree], batch["phyla_embeddings"], num_samples=1, dt_base=dt
            )
            
            avg_polytomy_sizes.append(avg_polytomy_size)
            num_polytomies_resolved.append(n_polytomies_resolved)

            sampled_tree = sampled_tree[0]
            num_topology_changes.append(n_topology_changes)
            avg_max_logits.append(avg_max_logit)
            if has_polytomy_fast(sampled_tree):
                sampled_tree = resolve_polytomies_random_deterministic(sampled_tree)
                if has_polytomy_fast(sampled_tree):
                    raise Exception(
                        "Whoa there is STILL a polytomy in the sampled tree, something is wrong!"
                    )
                num_polytomies += 1

            # Now do something with the sampled tree and the real trees
            sampled_trees.append(sampled_tree)
            if sanity_check:
                break

        if sanity_check:
            sampled_trees = sampled_trees*num_samples

        sampled = [
            number_to_name_newick(i, {int(i): v for i, v in mapping.items()}, True)
            for i in sampled_trees
        ]
        posterior_trees = [
            number_to_name_newick(i, {int(i): v for i, v in mapping.items()}, True)
            for i in real_trees
        ]
        starting_named = [
            number_to_name_newick(i, {int(i): v for i, v in mapping.items()}, True)
            for i in starting_trees_nw
        ]

        if save:
            import pickle

            with open(f"samples/sample_trees_{self.global_step}.pkl", "wb") as f:
                pickle.dump((sampled, posterior_trees), f)

        try:
            metrics = compare_likelihood_distributions(
                nexus_filepath, true_trees=posterior_trees, sampled_trees=sampled, threads=1
            )
        except Exception as e:
            print(f"Skipping likelihood calc due to error: {e}")
            metrics = {}

        metrics.update(
            kl_divergence_topological_distributions(
                posterior_trees, sampled, num_leaves=num_leaves
            )
        )
        metrics.update(
            split_bipartition_frequency_correlation(
                posterior_trees, sampled, num_leaves=num_leaves
            )
        )
        metrics.update(compare_branch_length_distributions(posterior_trees, sampled))

        rf_vals = []
        if len(posterior_trees) > 0 and len(sampled) > 0:
            for _ in range(100):
                t1 = random.choice(posterior_trees)
                t2 = random.choice(sampled)
                rf_vals.append(calculate_norm_rf(t1, t2))
        metrics["avg_posterior_sample_norm_rf"] = np.mean(rf_vals) if rf_vals else 0.0

        rf_paired = [calculate_norm_rf(s, e) for s, e in zip(starting_named, sampled)]
        metrics["start_avg_norm_rf"] = np.mean(rf_paired) if rf_paired else 0.0

        try:
            metrics.update(
                {
                    "start_" + k: v
                    for k, v in compare_likelihood_distributions(
                        nexus_filepath,
                        true_trees=starting_named,
                        sampled_trees=sampled,
                        threads=1,
                    ).items()
                }
            )
        except Exception:
            pass

        metrics.update(
            {
                "start_" + k: v
                for k, v in kl_divergence_topological_distributions(
                    starting_named, sampled, num_leaves=num_leaves
                ).items()
            }
        )
        metrics.update(
            {
                "start_" + k: v
                for k, v in split_bipartition_frequency_correlation(
                    starting_named, sampled, num_leaves=num_leaves
                ).items()
            }
        )
        metrics.update(
            {
                "start_" + k: v
                for k, v in compare_branch_length_distributions(
                    starting_named, sampled
                ).items()
            }
        )

        print(
            f"Num polytomies resolved in sampling: {num_polytomies} out of {num_samples}"
        )
        print(
            "Average topology changes during sampling: ", np.mean(num_topology_changes)
        )
        print("Average max logits during sampling: ", np.mean(avg_max_logits))
        overall_avg_polytomy_size = np.mean([s for s in avg_polytomy_sizes if s > 0]) if any(s > 0 for s in avg_polytomy_sizes) else 0.0
        print(f"Average polytomy size during sampling: {overall_avg_polytomy_size:.2f}")
        
        avg_num_polytomies_resolved = np.mean(num_polytomies_resolved)
        print(f"Average number of polytomies resolved during sampling: {avg_num_polytomies_resolved}")
        import pdb; pdb.set_trace()
        if self.record:
            wandb.log(
                {
                    "samples/number_of_polytomies_resolved": num_polytomies,
                    "samples/average_topology_changes": np.mean(num_topology_changes),
                    "samples/average_max_logits": np.mean(avg_max_logits),
                    "samples/average_num_polytomies_resolved": avg_num_polytomies_resolved,
                    "samples/average_polytomy_size": overall_avg_polytomy_size,
                },
                step=self.stepper,
            )

        return metrics

    def on_train_end(self):
        if self.record:
            wandb.finish()

    def training_step(self, batch, _):
        # Skip if batch is None (all items failed tokenization in collate_fn)
        if batch is None:
            logging.warning(
                "Skipping training step: batch is None (tokenization failed for all items)"
            )
            print(
                "Skipping training step: batch is None (tokenization failed for all items)"
            )
            return None

        # Increment stepper at the START to ensure all logs in this step use the same step number
        self.stepper += 1

        opt = self.optimizers()
        opt.zero_grad()

        if self.deepspeed:

            success = False
            num = 0
            failed = False

            # Logic, if we have an out of memmory error we just resample with a smaller subtree and rerun
            while not success:
                self.logger_.log(f"Entering step {num}", level=logging.INFO)
                error_tensor = torch.zeros(1).cuda()
                if num > 1:
                    self.logger_.log(
                        f"Batch is too large decreasing max tree size by a factor of 2 and num sequences",
                        level=logging.INFO,
                    )
                    if "loss" in locals():
                        loss = loss.detach()
                        del loss
                        gc.collect()

                    if num > 10:
                        return torch.tensor(0)

                    torch.cuda.empty_cache()
                    torch.distributed.barrier()
                    index, sub_tree_size, num_subtrees = self.dataset.chosen_tree

                    if sub_tree_size <= 5:
                        self.logger_.log(
                            f"We have reached the minimum tree size", level=logging.INFO
                        )
                        num_subtrees = 5
                        sub_tree_size = 5
                    elif num_subtrees > 100:
                        self.logger_.log(
                            f"Number of subtrees way too big {torch.distributed.get_rank()}",
                            level=logging.INFO,
                        )
                        num_subtrees = 50
                    else:
                        num_subtrees = int(num_subtrees // 2)
                        sub_tree_size = int(sub_tree_size // 2)

                        if sub_tree_size < 5:
                            sub_tree_size = 5
                        if num_subtrees < 1:
                            num_subtrees = 1

                    sub_batch = self.dataset.__getitem__(
                        index, preset_subtree_size=sub_tree_size
                    )
                    batch = self.dataset.collate_fn(
                        [sub_batch], preset_subtree_num=num_subtrees
                    )

                    # TODO: Run without adaptive batch size speedup
                    # TODO: Run with adaptive batch size speedup
                    if num <= 2:
                        new_max_aa = (
                            num_subtrees
                            * sub_tree_size
                            * self.dataset.return_max_length(self.dataset.name_to_seq)
                        )
                        self.logger_.log(
                            f"Updating the adaptive batch size sampler with this new information of the max aa of {new_max_aa}",
                            level=logging.INFO,
                        )
                        self.dataset.size_detector.update_max_aa(new_max_aa)

                    torch.distributed.barrier()
                    self.logger_.log(
                        f"We have all recreated our batches now moving on",
                        level=logging.INFO,
                    )
                try:
                    loss_status_tensor = torch.zeros(
                        torch.distributed.get_world_size()
                    ).cuda()
                    logs = self.step(batch)
                    if logs is not None:
                        loss = logs["loss"]
                        memory_error_tensor = torch.zeros(1).cuda()

                        # Go through every GPU get memmory used, if it is above 70% we will abort the manual backward and fail
                        stop_manual_backward = False
                        for i in range(torch.cuda.device_count()):
                            # Get the current memory usage
                            current_memory = torch.cuda.memory_allocated(i)
                            # Get the total memory
                            total_memory = torch.cuda.get_device_properties(
                                i
                            ).total_memory
                            fraction = current_memory / total_memory
                            if fraction > 0.75:
                                self.logger_.log(
                                    f"We detected that {i} device is above 75% memory usage!, will avoid manual backward!",
                                    level=logging.INFO,
                                )
                                stop_manual_backward = True
                                memory_error_tensor[0] = 1
                            self.logger_.log(
                                f"Device {i} is using {fraction} of its memory",
                                level=logging.INFO,
                            )
                            torch.distributed.barrier()

                        # If one at least fails the memory check then we will scuttle the backward
                        torch.distributed.all_reduce(memory_error_tensor)
                        if memory_error_tensor[0] > 0:
                            self.logger_.log(
                                f"Wow some is about to OOM we are scuttling the backward",
                                level=logging.INFO,
                            )
                            stop_manual_backward = True

                        # Okay what if one passes the memory check and still fails?
                        # loss_status_tensor = torch.zeros(1).cuda()

                        if not stop_manual_backward:
                            self.manual_backward(loss)
                            success = True
                            failed = False
                            self.logger_.log(f"Succeded!", level=logging.INFO)
                            loss_status_tensor[torch.distributed.get_rank()] = 1
                        else:
                            self.logger_.log(f"Skipping backward!", level=logging.INFO)
                            failed = True
                            success = False
                            num += 1
                            logs = None
                            loss_status_tensor[torch.distributed.get_rank()] = 1

                    else:
                        self.logger_.log(f"Failed!", level=logging.INFO)
                        num += 1
                        loss_status_tensor[torch.distributed.get_rank()] = 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger_.log(f"WARNING: out of memory", level=logging.INFO)
                        error_tensor[0] = 1
                        failed = True
                        logs = {"loss": torch.tensor(0)}
                        num += 1
                        loss_status_tensor[torch.distributed.get_rank()] = 1
                        self.logger_.log(f"Set up my status", level=logging.INFO)
                    else:
                        self.logger_.log(f"RAISING NEW ERROR {e}", level=logging.INFO)
                        raise e
                finally:
                    self.logger_.log(f"Entering check for the loss", level=logging.INFO)

                    while (
                        loss_status_tensor.sum() != torch.distributed.get_world_size()
                    ):
                        torch.distributed.all_reduce(loss_status_tensor)
                        self.logger_.log(
                            f"Waiting for everyone to finish\t{loss_status_tensor.sum()}\t{loss_status_tensor}",
                            level=logging.INFO,
                        )

                    torch.distributed.barrier()
                    torch.distributed.all_reduce(error_tensor)
                    if error_tensor[0] > 0:
                        self.logger_.log(
                            "Ooops someone had a OOM we should scuttle",
                            level=logging.INFO,
                        )
                        failed = True
                        success = False
                        num += 1

                    # print("Waiting")
                    torch.distributed.barrier()

                num += 1
                torch.distributed.barrier()
        else:
            success = False
            num = 0

            # Logic, if we have an out of memmory error we just resample with a smaller subtree and rerun
            while not success:

                # If fail will call zero grad again, may need this for deepspeed?
                opt.zero_grad()
                if num > 0:
                    logging.info(
                        "Batch is too large decreasing max tree and number of subtrees by a factor of 1.2"
                    )
                    index, sub_tree_size, num_subtrees = self.dataset.chosen_tree
                    new_sub_tree_size = sub_tree_size
                    new_num_subtrees = int(num_subtrees // 1.2)

                    if new_num_subtrees == 0:
                        new_num_subtrees = 1
                        new_sub_tree_size = int(sub_tree_size // 1.2)

                    if new_sub_tree_size < 5:
                        new_sub_tree_size = 5
                        new_num_subtrees = 1

                    if num <= 2:
                        new_max_aa = (
                            new_num_subtrees
                            * new_sub_tree_size
                            * self.dataset.return_max_length(self.dataset.name_to_seq)
                        )
                        logging.info(
                            f"Updating the adaptive batch size sampler with this new information of the max aa of {new_max_aa}"
                        )
                        self.dataset.size_detector.update_max_aa(new_max_aa)

                    if num > 10:
                        logging.info("We are spiraling, moving on")
                        return torch.tensor(0)

                    sub_batch = self.dataset.__getitem__(
                        index, preset_subtree_size=new_sub_tree_size
                    )
                    batch = self.dataset.collate_fn(
                        [sub_batch], preset_subtree_num=new_num_subtrees
                    )
                    logging.info(
                        f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                    )
                    logging.info(
                        f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                    )

                    gc.collect()
                try:
                    logging.info(
                        f"Memory allocated before step: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                    )
                    logging.info(
                        f"Memory reserved before step: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                    )

                    # --- HEAD 1: VELOCITY ---
                    logging.info("DEBUG: Starting Velocity Head Training")
                    logs_vel = self.step(batch, autoregressive=False)
                    loss_vel = logs_vel["loss"]
                    self.manual_backward(loss_vel)
                    self.clip_gradients(
                        opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                    )
                    opt.step()
                    opt.zero_grad()
                    logging.info("DEBUG: Finished Velocity Head Training")

                    del logs_vel
                    del loss_vel
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()

                    # --- HEAD 2: AUTOREGRESSIVE ---
                    logging.info("DEBUG: Starting Autoregressive Head Training")
                    logs = self.step(batch, autoregressive=True)
                    if "loss" not in logs:
                        import pickle

                        with open("debug_batch.pkl", "wb") as f:
                            pickle.dump(batch, f)
                        raise Exception(
                            "Loss not found in logs for autoregressive head!"
                        )
                    loss = logs["loss"]

                    logging.info(
                        f"Memory allocated before backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                    )
                    logging.info(
                        f"Memory reserved before backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                    )

                    self.manual_backward(loss)
                    self.clip_gradients(
                        opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                    )
                    opt.step()
                    opt.zero_grad()

                    success = True
                    failed = False

                    logging.info(
                        f"Memory allocated after backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                    )
                    logging.info(
                        f"Memory reserved after backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                    )

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.warning("WARNING: out of memory")
                        if hasattr(torch.cuda, "empty_cache"):
                            # Not sure about this
                            torch.cuda.empty_cache()

                        logging.info(
                            f"Memory allocated after OOM: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                        )
                        logging.info(
                            f"Memory reserved after OOM: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                        )

                        num += 1
                    else:
                        raise e

        # print(f"Entering a new world with status {failed}")
        if not failed and logs is not None:
            for k, v in logs.items():
                self.log(
                    k,
                    v.to("cuda"),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

            index, sub_tree_size, num_subtrees = self.dataset.chosen_tree
            lr = opt.optimizer.param_groups[0]["lr"]
            self.log("num_seq_per_subtree", sub_tree_size)
            logs["num_seq_per_subtree"] = sub_tree_size
            self.log("num_subtrees", num_subtrees)
            logs["num_subtrees"] = num_subtrees
            self.log("lr", lr)
            logs["lr"] = lr
            if self.logger_ is not None:
                self.logger_.log(logs, level=logging.INFO)
        else:
            print(logs)

        if logs is not None:
            if self.record:
                # wandb.log(logs)
                wandb.log(logs, step=self.stepper)
            if not self.dataset.msa_distance:
                self.dataset.update_normrf(logs["norm_rf_distance"])

            if self.deepspeed:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=1.0,  # tighten / loosen here
                    gradient_clip_algorithm="norm",
                )

            self.current_step_value += 1
            if self.deepspeed:
                opt.step()
            # print("Hi Im here waiting!")
            if self.deepspeed:
                torch.distributed.barrier()

            # Perform learning rate schedling
            if self.lr_scheduler == "cosine":
                sch1 = self.lr_schedulers()
                sch1.step()
            elif self.lr_scheduler == "cosine_warmup":
                sch1, sch2 = self.lr_schedulers()
                # Perform warmup
                if self.num_warmup_steps > 0:
                    sch1.step()
                    self.num_warmup_steps -= 1
                # Perform cosine annealing
                else:
                    sch2.step()
            elif self.lr_scheduler == "warmup":
                sch1 = self.lr_schedulers()
                # Perform warmup
                if self.num_warmup_steps > 0:
                    sch1.step()
                    self.num_warmup_steps -= 1

            # ADD CODE HERE TO UPDATE ADAPTIVE BATCH SIZE SAMPLER

            if self.global_step >= 500 and self.global_step % self.training_sampling_frequency == 0:
                # Moving to 1 sample so we can move faster
                metrics = self.sample_compare(
                    batch, train=True, num_samples=1, dt=self.dt
                )
                for k, v in metrics.items():
                    self.log(f"sample_metrics/{k}", v, on_step=True, logger=True)
                if self.record:
                    wandb.log(
                        {f"sample_metrics/{k}": v for k, v in metrics.items()},
                        step=self.stepper,
                    )
                print(metrics)

            return logs["loss"]
        else:
            return torch.tensor(0)

    def validation_step(self, batch, batch_idx):
        pass

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        norms = grad_norm(self, norm_type=2)
        if "grad_2.0_norm_total" in norms:
            total = norms["grad_2.0_norm_total"]
        else:
            total = norms.get("total_grad_norm", 0.0)  # hypothetical fallback
            if total == 0.0:
                # Just take the first key that looks like total if exists
                keys = [k for k in norms.keys() if "total" in k]
                if keys:
                    total = norms[keys[0]]

        # total = norms.get("grad_2.0_norm_total", 0.0)

        layer_norms = {k: v for k, v in norms.items() if "total" not in k}
        if layer_norms:
            max_grad = max(layer_norms.values())
            mean_grad = torch.mean(torch.stack(list(layer_norms.values())))
        else:
            max_grad = 0.0
            mean_grad = 0.0

        self.log("grad_norm_max", max_grad, prog_bar=True, on_step=True)
        self.log("grad_norm_mean", mean_grad, prog_bar=False, on_step=True)

        # Print a warning if exploding
        if max_grad > 1:
            print(
                f"[Warning] Gradient norm unusually high: max={max_grad:.2e}, mean={mean_grad:.2e}"
            )

        self.log("grad_norm_total", total)
        print(
            f"step {self.global_step:4d}  total_grad_norm = {total:.2f} mean is {mean_grad:.2f} max is {max_grad:.2f}"
        )
        if self.record:
            wandb.log(
                {
                    "grad/grad_norm_total": total,
                    "grad/grad_norm_max": max_grad,
                    "grad/grad_norm_mean": mean_grad,
                },
                step=self.stepper,
            )

    def configure_optimizers(self):
        if self.deepspeed:
            optimizer = FusedAdam(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        if self.lr_scheduler == "cosine":
            sch1 = CosineAnnealingLR(
                optimizer, T_max=self.num_annealing_steps
            )  # Set to current number of steps for training 7 days
            return [optimizer], [sch1]
        elif self.lr_scheduler == "cosine_warmup":
            sch1 = LinearLR(
                optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps
            )
            sch2 = CosineAnnealingLR(optimizer, T_max=self.num_annealing_steps)
            return [optimizer], [sch1, sch2]
        elif self.lr_scheduler == "warmup":
            sch1 = LinearLR(
                optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps
            )
            return [optimizer], [sch1]
        else:
            scheduler = []
            return optimizer
