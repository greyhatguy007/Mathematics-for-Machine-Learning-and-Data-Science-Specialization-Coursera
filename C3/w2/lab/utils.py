import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact,HBox, VBox
import matplotlib.gridspec as gridspec

df_anscombe = pd.read_csv('df_anscombe.csv')
df_datasaurus = pd.read_csv("datasaurus.csv")

def plot_anscombes_quartet():
    fig, axs = plt.subplots(2,2, figsize = (8,5), tight_layout = True)
    i = 1
    fig.suptitle("Anscombe's quartet", fontsize = 16)
    for line in axs:
        for ax in line:
            ax.scatter(df_anscombe[df_anscombe.group == i]['x'],df_anscombe[df_anscombe.group == i]['y'])
            ax.set_title(f'Group {i}')
            ax.set_ylim(2,15)
            ax.set_xlim(0,21)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            i+=1
        
def display_widget():

    dropdown_graph_1 = widgets.Dropdown(
    options=df_datasaurus.group.unique(),
    value='dino',
    description='Data set 1: ',
    disabled=False,
)
    
    statistics_graph_1 = widgets.Button(
    value=False,
    description='Compute stats',
    disabled=False,
    button_style='',
    tooltip='Description',
    icon='' 
)

    dropdown_graph_2 = widgets.Dropdown(
    options=df_datasaurus.group.unique(),
    value='h_lines',
    description='Data set 2: ',
    disabled=False,
)
    
    statistics_graph_2 = widgets.Button(
    value=False,
    description='Compute stats',
    disabled=False,
    button_style='',
    tooltip='Description',
    icon='' 
)
    plotted_stats_graph_1 = None
    plotted_stats_graph_2 = None

    fig = plt.figure(figsize = (8,4), tight_layout = True)
    gs = gridspec.GridSpec(2,2)
    ax_1 = fig.add_subplot(gs[0,0])
    ax_2 = fig.add_subplot(gs[1,0])
    ax_text_1 = fig.add_subplot(gs[0,1])
    ax_text_2 = fig.add_subplot(gs[1,1])
    df_group_1 = df_datasaurus.groupby('group').get_group('dino')
    df_group_2 = df_datasaurus.groupby('group').get_group('h_lines')
    sc_1 = ax_1.scatter(df_group_1['x'],df_group_1['y'], s = 4)
    sc_2 = ax_2.scatter(df_group_2['x'],df_group_2['y'], s = 4)
    ax_1.set_xlabel('x')
    ax_1.set_ylabel('y')
    ax_2.set_xlabel('x')
    ax_2.set_ylabel('y')
    ax_text_1.axis('off')
    ax_text_2.axis('off')
    
    def dropdown_choice(value, plotted_stats, ax_text, sc):
        if value.new != plotted_stats:
            ax_text.clear()
            ax_text.axis('off')
        sc.set_offsets(df_datasaurus.groupby('group').get_group(value.new)[['x', 'y']])
        fig.canvas.draw_idle()
    
        
    def get_stats(value, plotted_stats, ax_text, dropdown, val):
        value = dropdown.value
        if value == plotted_stats:
            return
        ax_text.clear()
        ax_text.axis('off')
        df_group = df_datasaurus.groupby('group').get_group(value)
        means = df_group.mean()
        var = df_group.var()
        corr = df_group.corr()
        ax_text.text(0,
                    0,
                    f"Statistics:\n      Mean x:      {means['x']:.2f}\n      Variance x: {var['x']:.2f}\n\n      Mean y:      {means['y']:.2f}\n      Variance y: {var['y']:.2f}\n\n      Correlation:  {corr['x']['y']:.2f}"
                    )
        if val == 1:
            plotted_stats_graph_1 = value
        if val == 2:
            plotted_stats_graph_2 = value
        
        
        

    dropdown_graph_1.observe(lambda value: dropdown_choice(value,plotted_stats_graph_1, ax_text_1, sc_1), names = 'value')
    statistics_graph_1.on_click(lambda value: get_stats(value, plotted_stats_graph_1, ax_text_1, dropdown_graph_1,1))
    dropdown_graph_2.observe(lambda value: dropdown_choice(value,plotted_stats_graph_2, ax_text_2, sc_2), names = 'value')
    statistics_graph_2.on_click(lambda value: get_stats(value, plotted_stats_graph_2, ax_text_2, dropdown_graph_2,2))    
    graph_1_box = HBox([dropdown_graph_1, statistics_graph_1])
    graph_2_box = HBox([dropdown_graph_2, statistics_graph_2])
    display(VBox([graph_1_box,graph_2_box]))
    

def plot_datasaurus():

    fig, axs = plt.subplots(6,2, figsize = (7,9), tight_layout = True)
    i = 0
    fig.suptitle("Datasaurus", fontsize = 16)
    for line in axs:
        for ax in line:
            if i > 12:
                ax.axis('off')
            else:
                group = df_datasaurus.group.unique()[i]
                ax.scatter(df_datasaurus[df_datasaurus.group == group]['x'],df_datasaurus[df_datasaurus.group == group]['y'], s = 4)
                ax.set_title(f'Group {group}')
                ax.set_ylim(-5,110)
                ax.set_xlim(10,110)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                i+=1

