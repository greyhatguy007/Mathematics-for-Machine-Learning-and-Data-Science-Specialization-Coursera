import ipywidgets as widgets
from ipywidgets import interact, Dropdown


question1 = "\033[1mSelect one of the options given:"
solution1 = {'A':'\033[91mNot quite. While both functions are used to create empty arrays, np.zeros() is initialized with the value 0.',
'B':"\033[91mNot quite. np.zeros() is inialized, and it gives an output of 0's.",
'C':"\033[91mNot quite. Most often, np.empty() is faster since it is not initialized.",
'D':''' \033[92mTrue! np.empty() creates an array with uninitialized elements from available memory space and may be faster to execute.'''}



def mcq(question, solution):
    s = ''
#     print(question)
    print("\033[1mPlease select the correct option:")
    answer_w = Dropdown(options = solution.keys(), value=None, layout=widgets.Layout(width='25%'))

    @interact(Answer = answer_w)
    def print_city(Answer):
        if(Answer != None):
            s = solution[Answer]
#             print("\n")
            print(s)
