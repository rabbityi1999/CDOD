# CDOD
This is the code of model CDOD, article "Continuous-Time and Discrete-Time Representation Learning for Origin-Destination Demand Prediction".
This paper has been accepted as a regular paper in the journal IEEE Transactions on Intelligent Transportation Systems (TITS).
Run on NewYork city
```
python main.py --data=NewYork --use_memory --divide_base=1500 --embedding_module=identity --node_dim=32 --memory_dim=64 --message_dim=64 --aggregator=hybrid
```
