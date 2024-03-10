import pickle
import gradio as gr


def load_data(data):
    id2preds = pickle.load(open(data, "rb"))

    ids = list(id2preds.keys())

    return gr.Dropdown(choices=ids), id2preds


def load_preds(id_, id2preds):
    return generate_html_table(tokens=id2preds[id_]["tokens"], scores=id2preds[id_]["scores"], categories=id2preds[id_]["categories"])

def generate_html_table(tokens, scores, categories):
    # Define the threshold for highlighting
    highlight_thresholds = [0.1, 0.25, 0.5, 0.75, 1.0]
    

    category_headers = "<th>" + "</th><th>".join(categories) + "</th>"

    # Start of the HTML string
    html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        .highest {
            background-color: #66ff33;
        }
        .medium {
            background-color: #ccffff;
        }
        .low {
            background-color: #ff99ff;
        }
        .tiny {
            background-color: #ffb3b3;
        }
        .zero {
            background-color: #ffffff;
        }
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border: 1px solid black;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>

<table>
    <tr>
        <th>Token</th>
        <|CATEGORY_HEADERS|>
    </tr>""".replace("<|CATEGORY_HEADERS|>", category_headers)
    
    # Loop through each token and its scores to create table rows
    for token, score in zip(tokens, scores):
        row = f"\n    <tr>\n        <td>{token}</td>"
        for s in score:
            # Apply high-score class based on the score value
            for i, threshold in enumerate(highlight_thresholds):
                if s <= threshold:
                    class_name = ["zero", "tiny", "low", "medium", "highest"][i]
                    break
            row += f"\n        <td class='{class_name}'>{s:.2f}</td>"
        row += "\n    </tr>"
        html += row
    
    # End of the HTML string
    html += """
</table>

</body>
</html>
"""
    return html


with gr.Blocks() as demo:

    id2preds = gr.JSON(visible=False) # holds data
    with gr.Row():
        data = gr.FileExplorer(label="Data")
        doc_id = gr.Dropdown(label="Doc ID")
        # error_type = gr.Dropdown(label="Error type", choices=["TP", "FP", "FN", "FNFP"])
        # random_button = gr.Button(label="Random")

    html = gr.HTML(label="Token Prediction Visualization")


    data.change(fn=load_data, inputs=data, output=[doc_id, id2preds])
    doc_id.change(fn=load_preds, inputs=[doc_id, id2preds,], output=[html])
    # random_button.change(fn=,output=[html, doc_id])