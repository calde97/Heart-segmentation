import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
from inference.inferences import load_model, run_inference
from inference.postprocessing import post_process
import nrrd
import threading

class FileSelectionApp:
    def __init__(self, root):
        self.root = root
        self.model_file = None
        self.patient_file = None
        self.destination_folder = None
        self.prediction_file = None
        self.destination_folder_post = None

        self.create_widgets()

    def create_widgets(self):
        self.root.title("Unet Inference")
        #self.root.geometry("400x400")
        self.root.config(bg="#f0f0f0")

        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.grid(row=0, column=0)

        # Add title label to title frame
        title_label = tk.Label(title_frame, text="Inference", font=("Arial", 16), bg="#f0f0f0")
        title_label.grid(row=0, column=0, pady=10)

        first_frame = tk.Frame(self.root, bg="#f0f0f0")
        first_frame.grid(row=1, column=0)

        model_button = tk.Button(first_frame, text="Browse Model", font=("Arial", 12), command=self.select_model_file)

        self.model_label = tk.Label(first_frame, text="None", font=("Arial", 12), bg="#f0f0f0", borderwidth=1, relief="solid")


        patient_button = tk.Button(first_frame, text="Browse Patient", font=("Arial", 12), command=self.select_patient_file)

        self.patient_label = tk.Label(first_frame, text="None", font=("Arial", 12), bg="#f0f0f0", borderwidth=1, relief="solid")

        # add button select destination folder
        self.destination_button = tk.Button(first_frame, text="Choose Dst Folder", font=("Arial", 12), command=self.select_destination_folder)

        # add label for destination folder
        self.destination_label = tk.Label(first_frame, text="None", font=("Arial", 12), bg="#f0f0f0", borderwidth=1, relief="solid")

        self.run_inference_button = tk.Button(first_frame, text="Run Inference", font=("Arial", 12), command=self.run_inference)

        self.progress = ttk.Progressbar(first_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')

        line_frame = tk.Frame(first_frame, height=2, bg="black")


        model_button.grid(row=0, column=0, padx=10, pady=10)
        self.model_label.grid(row=1, column=0, padx=10, pady=0)
        patient_button.grid(row=0, column=1, padx=10, pady=10)
        self.patient_label.grid(row=1, column=1, padx=10, pady=10)
        self.destination_button.grid(row=0, column=2, padx=10, pady=10)
        self.destination_label.grid(row=1, column=2, padx=10, pady=10)
        self.run_inference_button.grid(row=2, column=1, padx=10, pady=10)
        self.progress.grid(row=3, column=1, padx=10, pady=10)
        line_frame.grid(row=4, column=0, columnspan=100, sticky="ew")


        patient_frame = tk.Frame(self.root, bg="#f0f0f0")
        patient_frame.grid(row=2, column=0)

        label2 = tk.Label(patient_frame, text="PostProcessing", font=("Arial", 16), bg="#f0f0f0")
        label2.grid(row=0, column=0, pady=10)

        # add select file prediction
        self.prediction_button = tk.Button(patient_frame, text="Choose Prediction File", font=("Arial", 12),
                                           command=self.select_prediction_file)
        self.prediction_button.grid(row=1, column=0, padx=10, pady=10)

        # add label for prediction file
        self.prediction_label = tk.Label(patient_frame, text="None", font=("Arial", 12),
                                         bg="#f0f0f0", borderwidth=1, relief="solid")
        self.prediction_label.grid(row=2, column=0, padx=10, pady=10)

        # add button for selecting destination folder post processing
        self.destination_button_post = tk.Button(patient_frame, text="Choose Dst Folder", font=("Arial", 12),
                                                 command=self.select_destination_folder_post)
        self.destination_button_post.grid(row=3, column=0, padx=10, pady=10)

        # add label for destination folder post processing
        self.destination_label_post = tk.Label(patient_frame, text="No Destination Folder Selected", font=("Arial", 12),
                                               bg="#f0f0f0", borderwidth=1, relief="solid")
        self.destination_label_post.grid(row=4, column=0, padx=10, pady=10)

        # add button post processing
        self.post_processing_button = tk.Button(patient_frame, text="Run post Processing", font=("Arial", 12),
                                                command=self.run_post_processing)
        self.post_processing_button.grid(row=5, column=0, padx=10, pady=10)




    def select_model_file(self):
        self.model_file = filedialog.askopenfilename()

        if self.model_file:
            # get only the last part of the path
            last_part = self.model_file.split("/")[-1]
            self.model_label.config(text=last_part)


            try:
                self.model, self.device = load_model(self.model_file)
            except:
                messagebox.showerror("Error", "Could not load model file")
                self.model_label.config(text="No Model File Selected")
                self.model_file = None
        else:
            self.model_label.config(text="No Model File Selected")
    def select_patient_file(self):
        self.patient_file = filedialog.askdirectory()

        if self.patient_file:
            # get only the last 2 parts of the path
            last_part = self.patient_file.split("/")[-2:]
            self.patient_label.config(text=last_part)
        else:
            self.patient_label.config(text="No Patient File Selected")

    def select_destination_folder(self):
        self.destination_folder = filedialog.askdirectory()
        if self.destination_folder:
            last_part = self.destination_folder.split("/")[-1]
            self.destination_label.config(text=self.destination_folder)
        else:
            self.destination_label.config(text="No Destination Folder Selected")

    def run_inference_thread(self):
        #self.progress['value'] = 0
        self.progress.start()
        # run inference saves the output file in the destination folder and returns the patient
        self.patient = run_inference(self.model, self.patient_file, self.device, self.progress, self.destination_folder, self.root)
        self.progress.stop()
        messagebox.showinfo("Inference Finished", "Inference completed successfully!")

    def run_inference(self):

        if not self.model_file:
            messagebox.showerror("Error", "No model file selected")
            return

        if not self.patient_file:
            messagebox.showerror("Error", "No patient file selected")
            return

        if not self.destination_folder:
            messagebox.showerror("Error", "No destination folder selected")
            return

        inference_thread = threading.Thread(target=self.run_inference_thread)
        inference_thread.start()


    def select_prediction_file(self):
        self.prediction_file = filedialog.askopenfilename()

        if self.prediction_file:
            self.prediction_label.config(text="Prediction File: " + self.prediction_file)
        else:
            self.prediction_label.config(text="No Prediction File Selected")

    def select_destination_folder_post(self):
        self.destination_folder_post = filedialog.askdirectory()
        if self.destination_folder_post:
            self.destination_label_post.config(text="Destination Folder: " + self.destination_folder_post)
        else:
            self.destination_label_post.config(text="No Destination Folder Selected")

    def run_post_processing(self):
        if not self.prediction_file:
            messagebox.showerror("Error", "No prediction file selected")
            return

        print(self.prediction_file)

        if not self.destination_folder_post:
            messagebox.showerror("Error", "No destination folder selected")
            return


        try:
            output_post = post_process(self.prediction_file)
        except:
            messagebox.showerror("Error", "Could not run post processing")
            return
        # save output_post
        nrrd.write(self.destination_folder_post + "/prediction_post.nrrd", output_post)
        messagebox.showinfo("Post processing finished", "Completed successfully!")



root = tk.Tk()
app = FileSelectionApp(root)
root.mainloop()

