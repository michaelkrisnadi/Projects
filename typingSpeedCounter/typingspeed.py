import tkinter as tk
import time
import threading
import random


class TypeSpeedGUI:
    # Constructor
    def __init__(self):
        self.root = tk.Tk()
        # Window title
        self.root.title("Speed Typing Test")
        # Window size
        self.root.geometry("800x600")

        # List of texts from text.txt separated by new lines
        # https://www.kite.com/python/answers/how-to-read-a-newline-delimited-text-file-in-python#:~:text=Use%20str.,a%20newline%2Ddelimited%20text%20file&text=Call%20file.,each%20line%20in%20the%20file.
        self.texts = open("text.txt", "r").read().splitlines()

        # Creates frame
        self.frame = tk.Frame(self.root)

        # Displays random sentence from text.txt file above
        self.sample_label = tk.Label(self.frame, text=random.choice(self.texts), font=("Helvetica", 18))
        # sample_label location in the window row 0 column 0
        self.sample_label.grid(row=0, column=0, columnspan=2, padx=5, pady=10)

        # Entry box for us to type in (in row 1 or second row)
        self.input_entry = tk.Entry(self.frame, width=40, font=("Helvetica", 24))
        self.input_entry.grid(row=1, column=0, columnspan=2, padx=5, pady=10)
        # Triggering the timer to start when we start typing
        self.input_entry.bind("<KeyRelease>", self.start)  # https://www.tcl.tk/man/tcl8.4/TkCmd/bind.html

        # Speed counter on the second row (CPS, CPM, WPS, WPM)
        self.speed_label = tk.Label(self.frame, text="Speed: \n0.00 CPS\n0.00 CPM\n0.00 WPS\n0.00 WPM",
                                    font=("Helvetica", 18))
        self.speed_label.grid(row=2, column=0, columnspan=2, padx=5, pady=10)

        # Reset button, with command to reset (using self.reset function defined in line 86)
        self.reset_button = tk.Button(self.frame, text="Reset", command=self.reset, font=("Helvetica", 24))
        # Reset button location in the window
        self.reset_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

        # The frame is going to be packed and become expandable with expand=True
        self.frame.pack(expand=True)

        # Boolean for if the program is started or not
        self.counter = 0
        self.running = False

        self.root.mainloop()  # https://python-course.eu/tkinter_events_binds.php

    def start(self, event):  # When we bind key down to self.start its going to pass which key is pressed as well
        if not self.running:  # Start if not running
            if not event.keycode in [16, 17, 18]:  # If not shift, alt, and ctrl key
                self.running = True
                t = threading.Thread(target=self.time_thread)  # Time threading as the target, the Thread is running \
                # time_thread()
                # https://stackoverflow.com/questions/12435211/python-threading-timer-repeat-function-every-n-seconds/24488061
                t.start()
        # If there are mistakes in the typed text
        if not self.sample_label.cget('text').startswith(self.input_entry.get()):
            # Changes the foreground color (font) of the input to red
            self.input_entry.config(fg="red")
        else:
            # If there are no mistake
            self.input_entry.config(fg="black")
        # If finished typing and the text typed is exactly the same as the displayed text
        # https://stackoverflow.com/questions/6112482/how-to-get-the-tkinter-label-text/39381569
        if self.input_entry.get() == self.sample_label.cget('text'):
            # Stops running, stops time
            self.running = False
            # Foreground color of input entry is green (Indicating finished and correct typing)
            self.input_entry.config(fg="green")

    # Timing
    def time_thread(self):
        while self.running:
            # Every 0.1 second (https://stackoverflow.com/questions/510348/how-can-i-make-a-time-delay-in-python)
            time.sleep(0.1)
            self.counter += 0.1  # Adds 0.1 to the time every 0.1 second
            # CPS = The length of characters that we have in the input entry divided by the amount of seconds passed
            cps = len(self.input_entry.get()) / self.counter  # Characters per second
            cpm = cps * 60  # Characters per minute
            # Counting the WPS = characters entered, but split based on space as separator
            wps = len(self.input_entry.get().split(" ")) / self.counter  # Words per second
            wpm = wps * 60  # Word per minute
            self.speed_label.config(text=f"Speed: \n{cps:.2f} CPS\n{cpm:.2f} CPM\n{wps:.2f} WPS\n{wpm:.2f} WPM")
            # Changing the text, cps and cpm 2 decimal places

    def reset(self):
        self.running = False
        self.counter = 0
        self.speed_label.config(text="Speed: \n0.00 CPS\n0.00 CPM\n0.00 WPS\n0.00 WPM")
        self.sample_label.config(text=random.choice(self.texts))
        self.input_entry.delete(0, tk.END)


TypeSpeedGUI()
