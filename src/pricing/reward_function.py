import numpy as np


# Conversion rate curve:
# given the budget spent, returns number of clicks
def rewards(price_list):
    return price_list * np.exp(-price_list ** 1.8 / 700)