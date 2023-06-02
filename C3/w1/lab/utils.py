import numpy as np
from datetime import timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact_manual



class your_bday:
    def __init__(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        self.fig = fig
        self.ax = ax1
        self.ax_hist = ax2
        self.dates = [
            (date(2015, 1, 1) + timedelta(days=n)).strftime("%m-%d") for n in range(365)
        ]
        self.match = False
        self.bday_str = None
        self.bday_index = None
        self.n_students = 0
        self.history = []
        self.bday_picker = widgets.DatePicker(description="Pick your bday", disabled=False, style={'description_width': 'initial'})
        self.start_button = widgets.Button(description="Simulate!")

        display(self.bday_picker)
        display(self.start_button)

        self.start_button.on_click(self.on_button_clicked)

    def on_button_clicked(self, b):
        self.match = False
        self.n_students = 0

        self.get_bday()
        self.add_students()

    def get_bday(self):
        try:
            self.bday_str = self.bday_picker.value.strftime("%m-%d")
        except AttributeError:
            self.ax.set_title(f"Input a valid date and try again!")
            return
        self.bday_index = self.dates.index(self.bday_str)


    def generate_bday(self):
        # gen_bdays = np.random.randint(0, 365, (n_people))
        gen_bday = np.random.randint(0, 365)
        # if not np.isnan(self.y[gen_bday]):
        if gen_bday == self.bday_index:
            self.match = True
    
    def add_students(self):

        if not self.bday_str:
            return

        while True:
            if self.match:
                self.history.append(self.n_students)
#                 print(f"Match found. It took {self.n_students} students to get a match")
                n_runs = [i for i in range(len(self.history))]
                self.ax.scatter(n_runs, self.history)
                # counts, bins = np.histogram(self.history)
                # plt.stairs(counts, bins)
                # self.ax_hist.hist(bins[:-1], bins, weights=counts)
                self.ax_hist.clear()
                sns.histplot(data=self.history, ax=self.ax_hist, bins=16)
                # plt.show()
                break

            self.generate_bday()
            self.n_students += 1
            self.ax.set_title(f"Match found. It took {self.n_students} students.\nNumber of runs: {len(self.history)+1}")
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()


big_classroom_sizes = [*range(1,1000, 5)]
small_classroom_sizes = [*range(1, 80)]

def plot_simulated_probs(sim_probs, class_size):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
#     ax.scatter(class_size, sim_probs)
    sns.scatterplot(x=class_size, y=sim_probs, ax=ax, label="simulated probabilities")
    ax.set_ylabel("Simulated Probability")
    ax.set_xlabel("Classroom Size")
    ax.set_title("Probability vs Number of Students")
    ax.plot([0, max(class_size)], [0.5, 0.5], color='red', label="p = 0.5")
    ax.legend()
    plt.show()

    
    
class third_bday_problem:
    def __init__(self) -> None:
        fig, axes = plt.subplot_mosaic(
            [["top row", "top row"], ["bottom left", "bottom right"]], figsize=(10, 8)
        )
        self.fig = fig
        self.ax = axes["top row"]
        self.count_ax = axes["bottom left"]
        self.ax_hist = axes["bottom right"]
        self.ax.spines["top"].set_color("none")
        self.ax.spines["right"].set_color("none")
        self.ax.spines["left"].set_color("none")
        self.ax.get_yaxis().set_visible(False)
        x = np.arange(365)
        y = np.zeros((365,))
        y[y == 0] = np.nan

        y_match = np.zeros((365,))
        y_match[y_match == 0] = np.nan

        self.x = x
        self.y = y
        self.y_match = y_match
        self.match = False
        self.n_students = 0

        self.dates = [
            (date(2015, 1, 1) + timedelta(days=n)).strftime("%m-%d") for n in range(365)
        ]
        self.month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        self.history = []
        self.match_index = None
        self.match_str = None

        self.cpoint = self.fig.canvas.mpl_connect("button_press_event", self.on_button_clicked)

        # self.start_button = widgets.Button(description="Simulate!")

        # display(self.start_button)

        # self.start_button.on_click(self.on_button_clicked)

    def generate_bday(self):
        gen_bday = np.random.randint(0, 365)

        if not np.isnan(self.y[gen_bday]):
            self.match_index = gen_bday
            self.match_str = self.dates[gen_bday]
            self.y_match[gen_bday] = 1
            self.match = True

        self.y[gen_bday] = 0.5

    def on_button_clicked(self, event):
        if event.inaxes in [self.ax]:
            self.new_run()
            self.add_students()

    def add_students(self):

        while True:
            if self.match:
                self.history.append(self.n_students)
                n_runs = [i for i in range(len(self.history))]
                self.count_ax.scatter(n_runs, self.history)
                self.count_ax.set_ylabel("# of students")
                self.count_ax.set_xlabel("# of simulations")

                month_str = self.month_names[int(self.match_str.split("-")[0]) - 1]
                day_value = self.match_str.split("-")[1]
                self.ax.set_title(
                    f"Match found for {month_str} {day_value}\nIt took {self.n_students} students to get a match"
                )
                self.ax_hist.clear()
                sns.histplot(data=self.history, ax=self.ax_hist, bins="auto")
                break

            self.generate_bday()
            self.n_students += 1
            self.ax.set_title(f"Number of students: {self.n_students}")

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            if not np.isnan(self.y_match).all():
                markerline, stemlines, baseline = self.ax.stem(
                    self.x, self.y_match, markerfmt="*"
                )
                plt.setp(markerline, color="green")
                plt.setp(stemlines, "color", plt.getp(markerline, "color"))
                plt.setp(stemlines, "linestyle", "dotted")
            self.ax.stem(self.x, self.y, markerfmt="o")

    def new_run(self):
        y = np.zeros((365,))
        y[y == 0] = np.nan
        y_match = np.zeros((365,))
        y_match[y_match == 0] = np.nan
        self.y_match = y_match
        self.y = y
        self.n_students = 0
        self.match = False
        self.ax.clear()
