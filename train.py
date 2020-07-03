#getting experiment name
import sys
exp_param = sys.argv[0]

#loading dataset
from pycaret.datasets import get_data
data = get_data('juice')

#init setup
from pycaret.classification import setup, create_model, finalize_model
clf1 = setup(data, target = 'Purchase', logging=True, experiment_name = exp_param, session_id=123, silent=True, html=False)

#training rf
rf = create_model('rf')

#finalize rf
final_rf = finalize_model(rf)
