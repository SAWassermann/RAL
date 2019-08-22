# RAL
RAL - Reinforced stream-based Active Learning - is an active-learning technique relying on reinforcement-learning principles, using rewards and bandit-like algorithms.

In particular, the rewards are based on the usefulness of RAL's querying behaviour. The intuition behind the different reward values is that we attribute a positive reward in case RAL asks the oracle for ground truth and it was necessary (i.e. the underlying models would have predicted the wrong label), and a negative one otherwise, to penalize it (i.e. querying the oracle was unnecessary as the models predicted the right label anyway).

The system additionally makes use of the prediction certainty of the classification models. We combine the aforementioned reward mechanism
with the model's uncertainty to tune the sample-informativeness heuristic to better guide the query decisions.

For more details about RAL, please check out our papers [1, 2].

[1] *RAL – Improving Stream-Based Active Learning by Reinforcement Learning*  
S. Wassermann, T. Cuvelier, P. Casas  
in European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD), Workshop on Interactive Adaptive Learning (IAL), Würzburg, Germany, 2019.

[2] *ADAM & RAL: Adaptive Memory Learning and Reinforcement Active Learning for Network Monitoring*  
S. Wassermann, T. Cuvelier, P. Mulinka, P. Casas  
in International Conference on Network and Service Management (CNSM), Halifax, Canada, 2019.

Authors
-------
* **Sarah Wassermann** - [homepage](http://wassermann.lu)
* **Pedro Casas** - [homepage](http://pcasas.info/)
* **Thibaut Cuvelier** - [homepage](http://www.tcuvelier.be/)
