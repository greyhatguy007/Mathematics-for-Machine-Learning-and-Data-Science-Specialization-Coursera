import json
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact




def reset_answers():
    with open("answers.json", "w") as f:
        json.dump({}, f)

        
        
def exercise_1():
    mean = widgets.FloatText(
#         value='',
        placeholder=0.0,
        description='Mean:',
        disabled=False   
    )

    var = widgets.FloatText(
#         value='',
        placeholder=0.0,
        description='Variance:',
        disabled=False   
    )

    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(mean)
    display(var)
#     display(cov)
    
    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex1": {
                "mean": mean.value,
                "var": var.value,
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 1 saved.")
            

    button.on_click(on_button_clicked)

    
def exercise_2():
    hist = widgets.ToggleButtons(
        options=['left', 'center', 'right'],
        description='Your answer:',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
    )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(hist)

    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex2": {
                "hist": hist.value
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 2 saved.")
            

    button.on_click(on_button_clicked)
    
    
def exercise_3():
    sum_2_8 = widgets.FloatText(
#         value='',
        placeholder=0.0,
        description='P for sum=2|8',
        style = {'description_width': 'initial'},
        disabled=False   
    )

    sum_3_7 = widgets.FloatText(
#         value='',
        placeholder=0.0,
        description='P for sum=3|7:',
        style = {'description_width': 'initial'},
        disabled=False   
    )

    sum_4_6 = widgets.FloatText(
#         value='',
        placeholder=0.0,
        description='P for sum=4|6:',
        style = {'description_width': 'initial'},
        disabled=False   
    )
    
    sum_5 = widgets.FloatText(
#         value='',
        placeholder=0.0,
        description='P for sum=5:',
        style = {'description_width': 'initial'},
        disabled=False   
    )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(sum_2_8)
    display(sum_3_7)
    display(sum_4_6)
    display(sum_5)
    
    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex3": {
                "sum_2_8": sum_2_8.value,
                "sum_3_7": sum_3_7.value,
                "sum_4_6": sum_4_6.value,
                "sum_5": sum_5.value
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 3 saved.")
            

    button.on_click(on_button_clicked)

def exercise_4():
    mean = widgets.FloatText(
#         value='',
        placeholder=0.0,
        description='Mean:',
        disabled=False   
    )

    var = widgets.FloatText(
#         value='',
        placeholder=0.0,
        description='Variance:',
        disabled=False   
    )

    cov = widgets.FloatText(
#         value='',
        placeholder=0.0,
        description='Covariance:',
        disabled=False   
    )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(mean)
    display(var)
    display(cov)
    
    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex4": {
                "mean": mean.value,
                "var": var.value,
                "cov": cov.value
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 4 saved.")
            

    button.on_click(on_button_clicked)

    
def exercise_5():
    hist = widgets.ToggleButtons(
        options=['left', 'center', 'right'],
        description='Your answer:',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
    )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(hist)

    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex5": {
                "hist": hist.value
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 5 saved.")
            

    button.on_click(on_button_clicked)
    

def exercise_6():
    max_sum = widgets.IntSlider(
        value=2,
        min=2,
        max=12,
        step=1,
        description='Sum:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(max_sum)

    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex6": {
                "max_sum": max_sum.value
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 6 saved.")
            

    button.on_click(on_button_clicked)
    
    
def exercise_7():
    hist = widgets.ToggleButtons(
        options=['left-most', 'left-center', 'right-center', 'right-most'],
        description='Your answer:',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
    )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(hist)

    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex7": {
                "hist": hist.value
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 7 saved.")
            

    button.on_click(on_button_clicked)
    
    
def exercise_8():
    hist = widgets.ToggleButtons(
        options=['left-most', 'left-center', 'right-center', 'right-most'],
        description='Your answer:',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
    )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(hist)

    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex8": {
                "hist": hist.value
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 8 saved.")
            

    button.on_click(on_button_clicked)
    
    
def exercise_9():
    mean = widgets.ToggleButtons(
        options=['stays the same', 'increases', 'decreases'],
        description='The mean of the sum:',
        disabled=False,
        button_style='',
    )
    
    var = widgets.ToggleButtons(
        options=['stays the same', 'increases', 'decreases'],
        description='The variance of the sum:',
        disabled=False,
        button_style='',
    )
    
    cov = widgets.ToggleButtons(
        options=['stays the same', 'increases', 'decreases'],
        description='The covariance of the joint distribution:',
        disabled=False,
        button_style='',
    )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    print("As the number of sides in the die increases:")
    display(mean)
    display(var)
    display(cov)

    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex9": {
                "mean": mean.value,
                "var": var.value,
                "cov": cov.value,
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 9 saved.")
            

    button.on_click(on_button_clicked)
    
    

def exercise_10():
    options = widgets.RadioButtons(
                options=[
                    'the mean and variance is the same regardless of which side is loaded', 
                    'having the sides 3 or 4 loaded will yield a higher covariance than any other sides', 
                    'the mean will decrease as the value of the loaded side increases', 
                    'changing the loaded side from 1 to 6 will yield a higher mean but the same variance'
                ],
                layout={'width': 'max-content'}
            )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(options)

    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex10": {
                "options": options.value,
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 10 saved.")
            

    button.on_click(on_button_clicked)
    
    
def exercise_11():
    options = widgets.RadioButtons(
                options=[
                    'changing the direction of the inequality will change the sign of the covariance', 
                    'changing the direction of the inequality will change the sign of the mean', 
                    'changing the direction of the inequality does not affect the possible values of the sum', 
                    'covariance will always be equal to 0'
                ],
                layout={'width': 'max-content'}
            )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(options)

    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex11": {
                "options": options.value,
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 11 saved.")
            

    button.on_click(on_button_clicked)
    
    
def exercise_12():
    options = widgets.RadioButtons(
                options=[
                    'yes, but only if one of the sides is loaded', 
                    'no, regardless if the die is fair or not', 
                    'yes, but only if the die is fair', 
                    'yes, regardless if the die is fair or not'
                ],
                layout={'width': 'max-content'}
            )
    
    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()
    
    display(options)

    display(button, output)

    def on_button_clicked(b):
        
        with open("answers.json", "r") as f:
            source_dict = json.load(f)
            
        answer_dict = {
            "ex12": {
                "options": options.value,
            }
        }
        
        source_dict.update(answer_dict)
        
        with open("answers.json", "w") as f:
            json.dump(source_dict, f)
            
        with output:
            print("Answer for exercise 12 saved.")
            

    button.on_click(on_button_clicked)
    
    
def check_submissions():
    with open("./answers.json", "r") as f:
        answer_dict = json.load(f)
        
    saved_exercises = [k for k in answer_dict.keys()]
    expected = ['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6', 'ex7', 'ex8', 'ex9', 'ex10', 'ex11', 'ex12']
    missing = [e for e in expected if not e in saved_exercises]
    
    if missing:
        print(f"missing answers for exercises {[ex.split('ex')[1] for ex in missing]}\n\nSave your answers before submitting for grading!")
        return
    
    print("All answers saved, you can submit the assignment for grading!")