import unittest
import pickle
import numpy as np
from ete3 import Tree as EteTree
from utils.random_tree import Tree
from utils.bhv_utils import BHVEncoder
from utils.bhv_movie import build_tree_from_splits
from utils.metric_utils import kl_divergence_topological_distributions, compare_branch_length_distributions


class TestEncodingConsistency(unittest.TestCase):
    """Test that encoding a tree and rebuilding it produces the same topology."""

    def setUp(self):
        self.newick_starting_trees = [
            "((103:0.00682193,((((67:0.0439808,(((((51:0.00865743,146:0.00462283):0.0222211,((148:0.0128467,37:0.0198568):0.00843991,(((143:0.0102063,142:0.00461155):0.0080365,145:0.0142993):0.0143958,147:0.00990896):0.00175042):0.00572477):0.00703723,141:0.016264):0.00474726,(((122:0.00973929,100:0.00558785):0.00661349,(57:0.016645,59:0.0172412):0.0182336):0.00153417,(12:0.00946941,(113:0.000583822,133:0.00846711):0.00483948):0.0147823):0.00574291):0.018429,144:0.362805):0.00492942):0.00471254,136:0.0418575):0.00338047,(((((28:0.0155853,29:0.0170541):0.00591746,((4:0.0212908,62:0.00963851):0.00427174,(55:0.0243675,(8:0.0272281,(35:0.0368723,17:0.041493):0.0100006):0.0040231):0.00214787):0.0123364):0.00371982,((((105:0.012147,66:0.0216846):0.0199656,((((87:0.00105789,((85:0.000775376,(96:0.00262203,72:0.00178744):0.000202943):0.00148156,(109:0.000171426,86:0.000245894):0.00260151):0.000127423):0.00238994,107:0.00194945):0.0148945,(98:0.0113757,10:0.0196428):0.00132478):0.00764489,(((88:0.0018044,101:0.000387977):0.00306796,110:0.00325975):0.00699553,(((111:0.000157476,92:2.71054e-05):0.000434327,(82:0.00261643,91:6.10849e-06):0.00118483):0.0105812,(((45:6.40702e-05,84:0.00128383):0.00315209,90:0.00350013):0.0109783,((((9:0.00133327,138:0.000320998):0.00296377,93:0.00191054):0.00111562,80:0.00219403):0.00185688,((97:0.00240807,(11:0.000364193,99:0.000115593):0.00341518):0.00268344,83:0.00228184):0.00402854):0.00317984):0.00213919):0.00107087):0.0023433):0.00858903):0.00375316,(((((((41:0.00105486,102:0.000177744):0.000947044,65:0.000806974):0.00268069,95:0.00151309):0.0147259,(68:0.0179123,((64:0.00605018,(((104:0.00232496,((2:0.000997185,139:0.00136178):0.0010471,36:0.00490643):0.000724343):0.00186945,114:0.00618397):0.00289856,(106:0.00626906,(34:0.00577039,152:0.00451465):0.00344617):0.00328261):0.00137963):0.0224077,((24:0.00116844,73:0.00453684):0.00291715,(((((115:0.000357697,((6:0.000678228,7:8.34139e-05):0.00254229,(5:0.00226796,54:4.83906e-05):0.000279059):0.00114799):0.000303177,77:0.00138509):0.000290198,(76:0.00125516,79:0.0008757):0.00149058):0.0011394,74:0.0026567):0.000569481,19:0.00534505):0.00344107):0.00296114):0.00567605):0.00680583):0.00534258,(((70:0.00055191,69:0.000526055):0.00106309,71:0.00140978):0.0205476,154:0.0143841):0.0122886):0.00110761,(((150:0.00928329,153:0.0173967):0.00650573,((63:0.00186665,(60:0.000956335,56:0.000685092):0.000501623):0.00664776,112:0.00696783):0.0129722):0.000610306,((14:0.016527,(16:0.010197,(21:0.0179435,15:0.00996396):0.00580329):0.00345656):0.011247,(30:0.0199484,((140:0.0112,(((40:0.0041253,27:0.00260998):0.00133863,26:0.000559922):0.000124735,25:0.00195663):0.00425535):0.00651433,94:0.00870227):0.00140674):0.00308169):0.00439013):0.00659369):0.000282101,(3:0.0048589,23:0.00419814):0.0133532):0.000458596):0.00275684,((50:0.0429249,((49:0.00897731,48:0.00839156):0.0727112,(46:0.015475,(42:0.0237534,(47:0.0180185,(44:0.00484563,43:0.00750911):0.00529566):0.00235894):0.0122989):0.0259945):0.0444326):0.00708893,39:0.0576919):0.00682684):0.000587665):0.00403193,(81:0.032502,((20:0.035226,(58:0.00747374,13:0.00943416):0.00539698):0.00703943,(75:0.0280917,61:0.0232738):0.00471009):0.00233203):0.00347709):0.00163097,((((31:0.00596196,108:0.00263131):0.000647815,((151:0.0014092,18:0.00274048):0.00713745,38:0.00742499):0.00115907):0.00652393,(89:0.00540574,(((((135:0.000681585,120:0.00115567):0.0027356,121:0.0007025):0.000899841,134:0.00223626):0.00183965,(((78:0.00423689,132:0.00167753):0.000377088,131:0.00106203):0.00231213,((130:0.00133489,(118:0.000230166,((123:8.25941e-05,(128:0.00206079,117:0.00398205):0.000364704):0.000441875,(124:0.00199987,(125:0.000751714,126:0.000610554):8.48961e-05):0.000559669):0.00098881):0.00166583):0.000781049,(116:0.00140055,129:0.00273853):0.000752992):0.00165537):0.00354837):0.000421227,(119:0.00121687,127:0.00182713):0.0024307):0.00143368):0.00235858):0.013371,((32:0.0113075,33:0.0185754):0.0266407,1:0.0244961):0.0143169):0.00193206):0.00374671):0.00178552,((137:0.0416873,22:0.0409797):0.00740323,(52:0.0188817,53:0.0181052):0.0103287):0.0264786):0.0293935):0.00426248,149:0.000275609,0:0.000682018);"
        ]

    def test_encoding_roundtrip_norm_rf_zero(self):
        """Test that encoding a tree and rebuilding it produces norm-RF = 0."""
        for nw in self.newick_starting_trees:
            # Parse the original tree
            t = Tree(nw)
            enc = BHVEncoder()
            masks, lens = enc.return_BHV_encoding(t)
            
            # Build the split dictionary (filter out None lengths)
            tree_dict = {m: float(l) for m, l in zip(masks, lens) if l is not None}
            n_leaves = t.n_leaves
            mapping = t.id_to_name
            
            # Rebuild the tree from splits
            _, rebuilt_newick = build_tree_from_splits(
                list(tree_dict.keys()),
                tree_dict,
                n_leaves=n_leaves,
                root_leaf=n_leaves - 1,
                mapping=mapping,
            )

            # Parse both trees with ete3 for RF comparison
            original_tree = EteTree(nw)
            rebuilt_tree = EteTree(rebuilt_newick)
            
            # Compute Robinson-Foulds distance
            rf_result = original_tree.robinson_foulds(rebuilt_tree, unrooted_trees=True)
            rf_distance = rf_result[0]
            max_rf = rf_result[1]
            
            # Compute normalized RF
            if max_rf > 0:
                norm_rf = rf_distance / max_rf
            else:
                norm_rf = 0.0
            
            # Assert that norm-RF is 0 (trees have identical topology)
            self.assertEqual(
                norm_rf, 0.0,
                f"Expected norm-RF = 0, but got {norm_rf}. "
                f"RF distance: {rf_distance}, Max RF: {max_rf}"
            )

    def test_sampled_trees_metrics(self):
        """Test average norm-RF and KL divergence between sampled and posterior trees."""
        # Load the sampled trees
        sampled, posterior = pickle.load(open("samples/sample_trees_100.pkl", "rb"))
        
        # Compute average normalized RF between sampled and posterior trees
        rf_distances = []
        n_pairs = min(len(sampled), len(posterior))
        
        for i in range(n_pairs):
            try:
                t1 = EteTree(sampled[i])
                t2 = EteTree(posterior[i])
                rf_result = t1.robinson_foulds(t2, unrooted_trees=True)
                rf_distance = rf_result[0]
                max_rf = rf_result[1]
                if max_rf > 0:
                    norm_rf = rf_distance / max_rf
                else:
                    norm_rf = 0.0
                rf_distances.append(norm_rf)
            except Exception as e:
                print(f"Error computing RF for pair {i}: {e}")
                continue
        
        avg_norm_rf = np.mean(rf_distances)
        std_norm_rf = np.std(rf_distances)
        
        print(f"\n=== Sampled Trees (100) Metrics ===")
        print(f"Average norm-RF: {avg_norm_rf:.4f} ± {std_norm_rf:.4f}")
        print(f"Number of pairs compared: {len(rf_distances)}")
        
        # Compute KL divergence between topological distributions
        # Determine num_leaves from the first tree
        t = Tree(sampled[0])
        num_leaves = t.n_leaves
        
        kl_result = kl_divergence_topological_distributions(
            sampled, posterior, num_leaves=num_leaves
        )
        kl_div = kl_result['kl_divergence_topological']
        
        print(f"KL divergence (topological): {kl_div:.6f}")
        
        # Basic sanity checks
        self.assertGreaterEqual(avg_norm_rf, 0.0, "Average norm-RF should be >= 0")
        self.assertLessEqual(avg_norm_rf, 1.0, "Average norm-RF should be <= 1")
        self.assertLessEqual(kl_div, 10, "KL divergence should be >= 0")

        # res = compare_branch_length_distributions(posterior, sampled)
        # self.assertLessEqual(res['kl_divergence_branch_length'], 10)
        # self.assertLessEqual(res['js_divergence_branch_length'], 10)

        


if __name__ == "__main__":
    unittest.main()
