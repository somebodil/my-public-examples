import numpy as np

new_dictionary = {"Micheal": 18, "Elon": 17}
new_lis = list(new_dictionary.items())
con_arr = np.array(new_lis)
print("Convert dict to arr:", con_arr)
