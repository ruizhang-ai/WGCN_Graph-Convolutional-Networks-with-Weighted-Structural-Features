## WGCN
Graph structural information such as topologies or connectivities provides valuable guidance for graph convolutional networks (GCNs) to learn nodes' representations. Existing GCN models that capture nodes' structural information weight in- and out-neighbors equally or differentiate in- and out-neighbors globally without considering nodes' local topologies. We observe that in- and out-neighbors contribute differently for nodes with different local topologies. To explore the directional structural information for different nodes, we propose a GCN model with weighted structural features, named WGCN. WGCN first captures nodes' structural fingerprints via a direction and degree aware Random Walk with Restart algorithm, where the walk is guided by both edge direction and nodes' in- and out-degrees. Then, the interactions between nodes' structural fingerprints are used as the weighted \textit{node structural features}. To further capture nodes' high-order dependencies and graph geometry, WGCN embeds graphs into a latent space to obtain nodes' latent neighbors and geometrical relationships. Based on nodes' geometrical relationships in the latent space, WGCN differentiates latent, in-, and out-neighbors with an attention-based geometrical aggregation. Experiments on transductive node classification tasks show that WGCN outperforms the baseline models consistently by up to 17.07\% in terms of accuracy on five benchmark datasets. More about WGCN is described in [*[Yunxiang Zhao, Jianzhong Qi, Qingwei Liu and Rui Zhang. "WGCN: Graph Convolutional Networks with Weighted Structural Features". arXiv:2104.14060 (2021)]*](https://arxiv.org/abs/2104.14060).

### Required Packages
Please refer requirements.txt. To install the missing packages, run the following command:
```
pip install -r requirements.txt
```

### Run the demo
Please first download two feature files required for the squirrel and the crocodile datasets from:
```
link: https://pan.baidu.com/s/1E3sw3YoQFqoNpH_KzOz_1w 
password: 7f3x
```
Then put the two files to "./data/squirrel/" and "./data/crocodile/", respectively. 

To replicate the results in Table 2, run the following command in the source directory:
```
>> bash run_chameleon.txt
>> bash run_squirrel.txt
>> bash run_crocodile.txt
>> bash run_cora_ml.txt
>> bash run_citeseer_ml.txt
```
Results will be stored in the "results" folder. We recommend to use the test accuracy of 1000 epoch.

### Citation
Please cite our paper if you use this code or datasets in your own work:
```
@inproceedings{zhao2021wgcn,
  title={WGCN: Graph Convolutional Networks with Weighted Structural Features},
  author={Zhao, Yunxiang and Qi, Jianzhong and Liu, Qingwei and Zhang, Rui},
  booktitle={ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2021}
}```
