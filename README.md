# SensorFormer

1. synthetic_data is the source code for Sec. II
2. SensorFormer is the source code and data for our SensorFormer model
	* one2one-w-val.py: one2one calibration scheme
	* many2one-w-val.py: many2one calibration scheme
	* many2many-w-val.py: many2many calibration scheme
	* new_data/ : data used for the model
	* result/ : results 
	* loss: loss functions used for the model
	* attention_map: attention map data used for the experiment section
3. SensorFormer_Lite is the source code and data for SensorFormer Lite model
	* many2many_lite_random.py: input downsampling by random selection
	* many2many_lite_mean.py: input downsampling by using the mean value
	* many2many_lite_soft.py: input downsampling by weighted embeddings
	* new_data/ : data used for the model
	* result/ : results 
	* loss: loss functions used for the model
	* attention_lite: attention map data used for the experiment section
4. Arduino_model shows the code and steps to convert the trained model to code running on Arduino. 