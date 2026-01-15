from utils.bhv_distance import bhv_geodesic_with_support
from utils.random_tree import Tree
import logging 
logger = logging.getLogger(__name__)
import random as rm

class BHVEncoder():

    def _choose_root(self, tree, root=None):
        """Choose a root for an (unrooted) tree.

        If `root` is provided, use it. Otherwise, pick a random leaf in 1..n.
        Falls back to the smallest node id if no leaves are found.
        """
        if root is not None:
            return root

        # Prefer a random leaf among 1..n_leaves
        leaves = [u for u in tree.adj if len(tree.adj[u]) == 2]
        if len(leaves) > 1:
            #THIS SHOULD NOT HAPPEN
            import pdb; pdb.set_trace()
        else:
            return leaves[0]

    def compute_edge_masks(self, tree, root=None):
        """
        Returns:
        edge_masks: dict[(u,v)] -> mask over leaves below v, for directed edges u->v
                    Only for edges that correspond to nontrivial splits.
        Assumes leaves are labeled 1..n_leaves, internal nodes >= n_leaves.
        """
        #root = self._choose_root(tree, root)
        root = tree.root
        n = tree.n_leaves
        full = (1 << n) - 1

        parent = {}
        order = []

        # iterative DFS to get postorder
        stack = [root]
        parent[root] = None
        while stack:
            u = stack.pop()
            order.append(u)
            for v in tree.adj[u]:
                if v not in parent:
                    parent[v] = u
                    stack.append(v)

        # postorder accumulation of leaf masks
        #So every leaf has a bit mask associated with just that identity from 0 to n
        #Every internal node then gets ORd with that bit mask to make a bitmask that represents all nodes it is a parent of

        node_mask = {u: 0 for u in tree.adj}
        for u in reversed(order):
            if 0 <= u < n:  # leaf
                node_mask[u] = (1 << u)
            else:
                m = 0
                for v in tree.adj[u]:
                    if parent.get(v) == u:  # child
                        m |= node_mask[v]
                node_mask[u] = m

        # build edge masks for internal edges
        # For each internal edge take the sort of anti-not somehow like if both 1 then 0, this represents the split of the tree
        edge_masks = []
        edge_lengths = []
        for v in tree.adj:
            p = parent.get(v)
            if p is not None:
                A = node_mask[v]
                if A != 0 and A != full:  # nontrivial split
                    # canonical side
                    Ac = full ^ A
                    canon = min(A, Ac)
                    edge_masks.append(canon)
                    edge_lengths.append(tree.length(p,v))

        return edge_masks, edge_lengths

    def return_BHV_encoding(self, tree):
        #Find root of the tree
        
        edge_masks, edge_lengths = self.compute_edge_masks(tree)
        return edge_masks, edge_lengths
    
    def BHV_length(self, one, two):
        edge_mask_1, edge_length_1 = one 
        edge_mask_2, edge_length_2 = two 

        t1 = {x:y for x,y in zip(edge_mask_1, edge_length_1)}
        t2 = {x:y for x,y in zip(edge_mask_2, edge_length_2)}
        result = bhv_geodesic_with_support(t1, t2, n_leaves=self.n_leaves)

        print("BHV distance:", result["distance"])
        for i, seg in enumerate(result["segments"]):
            print(f"Segment {i}:")
            print("  Ai (collapsed):", seg["Ai"])
            print("  Bi (grown):    ", seg["Bi"])
            print("  ratio:", seg["ratio"])
            # seg["start_splits"], seg["end_splits"] give you orthant topology at each step

def best_decomposition_for_any_orientation(split, sorted_components, full):
    best = None
    best_split = None

    for cand in (split, full ^ split):
        for comps in sorted_components:
            region_mask = 0
            for comp in comps:
                region_mask |= comp

            if cand & ~region_mask:
                continue

            relevant = []
            ok = True
            for comp in comps:
                inter = comp & cand
                if inter == 0:
                    continue
                if inter == comp:
                    relevant.append(comp)
                else:
                    ok = False
                    break

            if ok and (best is None or len(relevant) < len(best)):
                best = relevant
                best_split = cand

    return best_split, best



def return_sampled_tree_boundary_decisions(newick_tree_one, newick_tree_two, verbose = False):
    from utils.utils import find_polytomy_nodes, polytomy_components_at_node, bucket_by_overlap
    from utils.bhv_movie import build_tree_from_splits
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    t1 = Tree(newick_tree_one)
    t2 = Tree(newick_tree_two)

    enc = BHVEncoder()
    t1_edge_mask, t1_edge_length = enc.return_BHV_encoding(t1)
    t2_edge_mask, t2_edge_length = enc.return_BHV_encoding(t2)

    tree1 = {m: l for m, l in zip(t1_edge_mask, t1_edge_length)}
    tree2 = {m: l for m, l in zip(t2_edge_mask, t2_edge_length)}
    
    geodesic_result = bhv_geodesic_with_support(tree1, tree2, n_leaves=t1.n_leaves)
    segments = geodesic_result['segments']
    idxs = [rm.randrange(0, len(segments)-1)]

    final_labels = []
    full = (1 << t1.n_leaves) - 1

    for bi in idxs:
        lengths = {m:L for m, L in segments[bi]['end_lengths'].items() if L > 1e-8}
        Bi_splits = segments[bi]['Bi'].copy()

        #canonicalize Bi_splits 
        Bi_splits = [min(split, full ^ split) for split in Bi_splits]

        clustered_buckets = bucket_by_overlap(Bi_splits)
        num_iter = 0    

        while clustered_buckets:
            G, newick = build_tree_from_splits(list(lengths.keys()), lengths, t1.n_leaves, root_leaf=t1.n_leaves-1, mapping=t1.id_to_name)
            nodes_to_explore = find_polytomy_nodes(G, min_degree=4)
            components_to_fix = [polytomy_components_at_node(G, i, t1.n_leaves) for i in nodes_to_explore]
            sorted_components = sorted(components_to_fix, key=lambda comps: min(c.bit_count() for c in comps))
            logger.debug(f"Boundary index {bi}: exploring {len(nodes_to_explore)} polytomy nodes")

            Bi_split_component = {}

            for split in Bi_splits:
                logger.debug(f"Looking for relevant components found for {split}:")
                if split not in Bi_split_component:
                    Bi_split_component[split] = []

                use_split, best = best_decomposition_for_any_orientation(split, sorted_components, full)

                if best is None:
                    if len(final_labels) > 10:
                        logger.debug(" Could not find relevant components for split, but have previous labels, skipping this split")
                        return final_labels
                    else:
                        raise Exception("Could not find relevant components for split and no final labels so this is a huge error!")
               
                logger.debug(f" BEST RELEVANT component found for {split}:")
                for comp in best:
                    logger.debug(f"BEST RELEVANT component: {[i for i in range(comp.bit_length()) if (comp >> i) & 1]}")
                logger.debug("\n\n\n\n\n\n\n")

                if use_split != split:
                    logger.debug(f"Using complementary orientation for split {split} -> {use_split}")
                    if use_split != split:
                        Bi_splits.remove(split)
                        Bi_splits.append(use_split)  # or manage as a set (recommended)

                        # also fix clustered_buckets if you keep them:
                        for cluster in clustered_buckets:
                            if split in cluster:
                                cluster.remove(split)
                                cluster.add(use_split)
                
                Bi_split_component[use_split] = best

            #Okay for each clustered bucket, find the smallest group we need to merge, and add that to the merge
            initial_labels = []
            for cluster in clustered_buckets:
                to_add = None 
                smallest_num = None

                for potential_split in cluster:
                    sub_split = Bi_split_component[potential_split]
                    if smallest_num is None or len(sub_split) < smallest_num:
                        smallest_num = len(sub_split)
                        to_add = (potential_split, sub_split)

                if to_add is not None:
                    initial_labels.append(to_add)
                else:
                    raise Exception("Could not find any merges for this cluster, something is wrong!")

            final_labels.append({'newick': newick, 'labels': initial_labels})

            #Now merge those components but only do 1 at a time so we keep the labels but just do 1 to avoid weird errors 
            for split_merge, comps in initial_labels:

                check = comps[0]
                for c in comps[1:]:
                    check |= c
                if check != split_merge:
                    raise Exception("Merged components do not equal original split, something is wrong!")

                lengths[split_merge] = 0.1  #length really does not matter

                found = False
                #Remove this merged mask from clustered_buckets
                for cluster in clustered_buckets:
                    if split_merge in cluster:
                        cluster.remove(split_merge)
                        found = True
                
                Bi_splits.remove(split_merge)
                logger.debug(f"Removed {split_merge} from Bi_splits")
                
                if not found:
                    raise Exception("Could not find merged mask in clustered buckets which should not be possible")
            
            #Remove any empty clusters from clustered buckets
            clustered_buckets = [cluster for cluster in clustered_buckets if cluster]
            num_iter += 1
            if not initial_labels:
                print("Nothing happened this iteration??")
                import pdb; pdb.set_trace()
            if num_iter > 20 and not initial_labels:
                import pdb; pdb.set_trace()
                raise Exception("Too many iterations trying to resolve boundary decisions, something is wrong")
            if clustered_buckets and not initial_labels:
                for original_births in segments[bi]['Bi']:
                    logger.debug(f"Original Bi split: {[i for i in range(original_births.bit_length()) if (original_births >> i) & 1]} or {original_births}")
                import pdb; pdb.set_trace()
                raise Exception("Stuck with unresolved clustered buckets but no possible merges, something is wrong")
        
    return final_labels
    

        

            

        

        



def return_sampled_tree_orthant_velocity(newick_tree_one, newick_tree_two, time_point):
    from utils.bhv_movie import sample_tree_along_geodesic
    t1 = Tree(newick_tree_one)
    t2 = Tree(newick_tree_two)

    enc = BHVEncoder()
    t1_edge_mask, t1_edge_length = enc.return_BHV_encoding(t1)
    t2_edge_mask, t2_edge_length = enc.return_BHV_encoding(t2)

    tree1 = {m: l for m, l in zip(t1_edge_mask, t1_edge_length)}
    tree2 = {m: l for m, l in zip(t2_edge_mask, t2_edge_length)}
    geodesic_result = bhv_geodesic_with_support(tree1, tree2, n_leaves=t1.n_leaves)
    G, newick, info = sample_tree_along_geodesic(geodesic_result, t1.n_leaves, u=time_point)

    #This was debugging for a particular tree
    # # newick = '(((((((((((((((((((((((26:0.0,27:0.0,28:0.0,29:0.0):0.365231256756337,30:0.0,31:0.0):0.46562144283500395,32:0.0):0.13098585901031182,(33:0.0,34:0.0):0.5291394654659506):0.11776353850130768,35:0.0):0.26157583756205705,(24:0.0,25:0.0):0.36258921686734963):0.5033803018034739,36:0.0):0.20237942744692658,37:0.0):0.3963250321841045,38:0.0):0.3885226443458947,39:0.0):0.5347065021413112,40:0.0):0.1889407459500561,((41:0.0,42:0.0):0.3123941602336278,43:0.0):0.4617405890461322):0.4096061346588971,44:0.0):0.09230307198090551,45:0.0):0.3418153243061678,46:0.0):0.3554843469643267,47:0.0):0.10002600125544758,48:0.0,49:0.0):0.4654256573541056,(((16:0.0,17:0.0,18:0.0):0.0738641626933164,19:0.0):0.31399054621088424,(20:0.0,21:0.0,22:0.0):0.07592098916706257,23:0.0):0.3001943741954219):0.24382687114231433,((((((((((50:0.0,51:0.0):0.22142174476410542,52:0.0):0.17074685221592145,53:0.0):0.22838704277064867,(54:0.0,55:0.0,56:0.0):0.37569841866156367):0.13616650211948078,57:0.0):0.0641547613661573,58:0.0):0.5359239417814602,59:0.0):0.25989987338680104,((60:0.0,61:0.0,62:0.0):0.43040397244018275,(63:0.0,64:0.0):0.39425131617022474,65:0.0,66:0.0):0.5344297477988371):0.22352185160016247,67:0.0,68:0.0):0.5166211118827367,69:0.0,70:0.0):0.43780235312355326):0.4562863329696488,71:0.0):0.44293449831375736,72:0.0):0.15731748844500548,((((((((0:0.0,1:0.0):0.16131819778555037,2:0.0):0.1598279400172913,3:0.0):0.07150417342001554,4:0.0):0.4143589617385931,5:0.0):0.18956447983105795,6:0.0):0.26457093652387426,7:0.0):0.3509120493508661,(((((8:0.0,9:0.0,10:0.0):0.23219967414249623,11:0.0):0.47890150538212484,(12:0.0,13:0.0):0.5300847583428087):0.19037701078440555,14:0.0):0.4826082684585838,15:0.0):0.21491917547287828):0.24452064127567635):0.3516082338078238,(((((((((((((((((((((100:0.0,101:0.0):0.4075151593913468,98:0.0,99:0.0):0.35246140233112444,(102:0.0,103:0.0):0.31257845845204657,104:0.0):0.13875849332367882,105:0.0):0.3956641634593137,(106:0.0,107:0.0):0.3082347174751331):0.4173298184039069,108:0.0):0.1890248480912928,(96:0.0,97:0.0):0.4099729156249853):0.26452779741123134,109:0.0):0.4070499942463294,110:0.0):0.11205970478777688,(111:0.0,112:0.0):0.2561745153331085):0.49159967584084857,(((91:0.0,92:0.0):0.08362457977610852,89:0.0,90:0.0,93:0.0,94:0.0):0.18330279434802157,95:0.0):0.3493371033744729):0.3663038054950591,113:0.0):0.23649380819270865,114:0.0):0.4464045694170972,115:0.0):0.5178991085420517,(((((116:0.0,117:0.0):0.1626560063932636,118:0.0):0.22208903780237932,119:0.0):0.3658530512957051,120:0.0):0.1166761718242802,(121:0.0,122:0.0):0.32511291828552646):0.5020210696622494):0.07262859964821185,(((((((((81:0.0,82:0.0):0.4701788003557065,83:0.0):0.2838378314914868,84:0.0):0.36065683669899545,85:0.0):0.12577566798746628,86:0.0):0.327628012142499,87:0.0):0.11414385983118146,88:0.0):0.10439351856711432,78:0.0,79:0.0,80:0.0):0.2883345348574025,((((73:0.0,74:0.0):0.16959541461333827,75:0.0):0.3470113937468134,76:0.0):0.15024189043096217,77:0.0):0.3342242085245652):0.47486736798186163):0.4584589852726523,123:0.0):0.12823236347459735,124:0.0):0.4647828243748471,125:0.0):0.23519184935769752,(126:0.0,127:0.0):0.20818437562919903):0.22921477851937236,((((131:0.0,132:0.0,133:0.0):0.49818678251290305,(128:0.0,129:0.0,130:0.0):0.1081456116685175,134:0.0):0.374861791247522,135:0.0):0.0625760728935306,136:0.0):0.176919171345831):0.4956524440616365,((((((((139:0.0,140:0.0,141:0.0):0.2044406361563448,(137:0.0,138:0.0):0.24757861246008647):0.36496011409578083,142:0.0):0.3394472667518442,143:0.0):0.056707772310875176,144:0.0):0.15478537361656458,((((145:0.0,146:0.0):0.32455724511050016,147:0.0):0.15788405331664848,148:0.0):0.11136587699554824,149:0.0):0.4129118538256446):0.4299676167090581,150:0.0):0.07132799911059619,(((151:0.0,152:0.0):0.35895846909210893,153:0.0):0.15181817937207243,154:0.0):0.4425292371341138):0.24829617861092312);'
    # test = Tree(newick)
    # edge_mask, edge_length = enc.return_BHV_encoding(test)
    # active_edge_mask = []
    # active_edge_length = []
    # for i,z in zip(edge_mask, edge_length):
    #     if z > 1e-6:
    #         active_edge_mask.append(i)
    #         active_edge_length.append(z)
    # print(f"Sampled tree has {len(edge_mask)} edges, velocity has {len(info['velocity'])} entries")
    # real_max_bit = max(m.bit_length() for m in active_edge_mask)
    # for i in info['active_velocity']:
    #     vel = i
    #     if vel.bit_length() == real_max_bit+1:
    #         vel = remove_bit(vel, t1.n_leaves+1)
    #         print("Adjusted velocity bitmask by removing dummy leaf bit.")
    #     if vel not in active_edge_mask:
    #         print(f"Velocity entry {vel} not found in sampled tree edge masks!")
    #         print(bin(vel))
    #         print([i for i in range(vel.bit_length()) if (vel >> i) & 1])
    #         print([[i for i in range(vel.bit_length()) if (vel >> i) & 1] for vel in active_edge_mask])
    #         import pdb; pdb.set_trace()
    #     else:
    #         print("FOUND WOOHOOOO")
    # import pdb; pdb.set_trace()

    return newick, info['active_velocity']


def test_bhv_on_two_random_20_leaf_trees():
    from utils.bhv_movie import make_bhv_topology_movie
    n = 20
    print("Generating random trees...")
    T1 = Tree(num_leaves=n, random=True)
    T2 = Tree(num_leaves=n, random=True)

    print("Tree 1 Newick:", T1)
    print("Tree 2 Newick:", T2)

    enc = BHVEncoder()

    print("Encoding trees into bitmask form...")
    edge_masks_1, edge_lengths_1 = enc.return_BHV_encoding(T1)
    edge_masks_2, edge_lengths_2 = enc.return_BHV_encoding(T2)

    tree1 = {m: l for m, l in zip(edge_masks_1, edge_lengths_1)}
    tree2 = {m: l for m, l in zip(edge_masks_2, edge_lengths_2)}

    print("Computing BHV geodesic with support pairs...")
    result = bhv_geodesic_with_support(tree1, tree2, n_leaves=T1.n_leaves)

    print("\n======================")
    print("BHV DISTANCE =", result["distance"])
    print("======================\n")

    print("Common-edge contribution squared =", result["common_sq"])
    print("Disjoint-edge contribution squared =", result["disjoint_sq"])
    print("Number of support pairs =", len(result["A_support"]))
    print()

    for i, seg in enumerate(result["segments"]):
        print(f"--- Segment {i} ---")
        print("Ai (collapse):", seg["Ai"])
        print("Bi (grow):    ", seg["Bi"])
        print("||A||=", seg["normA"], "||B||=", seg["normB"], "ratio=", seg["ratio"])
        print("#start splits =", len(seg["start_splits"]))
        print("#end splits   =", len(seg["end_splits"]))
        print()

    print("Test completed.")

    make_bhv_topology_movie(
        result,
        n_leaves=T1.n_leaves,
        root = T1.n_leaves-1,
        filename="bhv_topology_20leaf.gif",
        mapping=T1.id_to_name,
        F=20,
        fps=1,   # 1 frame per second (one per step)
    )


##############################################################################
# Run the test
##############################################################################

if __name__ == "__main__":
    test_bhv_on_two_random_20_leaf_trees()

        
       
        
