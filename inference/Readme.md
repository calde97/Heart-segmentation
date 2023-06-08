### GUI for inference

In order to to inference you have to run the following command if you are in the root of the project:

```bash
python inference/gui.py
```

If you are in the `inference` folder you can run the following command:

```bash
python gui.py
```

After that the GUI will open. You can select the model you want to use for inference 
the patient and the path to the image you want to segment. After that you can press the `Run Inference` button .
You can also do the post processing

### Dash Visualization

In order to run the dash visualization you have to run the following command if you are in the root of the project:

```bash
python inference/visualization_results.py
```

After that you can see from the terminal the link to the dash visualization.

You should put the results of the inference inside the `inference/data_to_visualize` folder.