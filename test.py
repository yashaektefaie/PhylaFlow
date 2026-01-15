from ete3 import Tree

#newick_string = "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);"
newick_string = "((((2:0.5106,(3:0.7864,4:0.2627):0.3287):0.5470):0.6684,5:0.6891):0.1975);"
tree = Tree(newick_string)

node_counter = 0
for node in tree.traverse():
    if not node.name: # Only name nodes that don't already have one
       node.name = f"Internal_{node_counter}"
       node_counter += 1

edges = []
for node in tree.traverse():
    if node.up:  # Check if it's not the root
        edges.append((node.up.name if node.up.name else "InternalNode", 
                      node.name if node.name else "InternalNode"))
    else:
        edges.append(('ROOT', node.name if node.name else "InternalNode"))  # Root node

print(edges)
