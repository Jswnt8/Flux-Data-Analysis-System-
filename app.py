"""
Flux - Data Analysis System
-----------------------------------
A Python-based system for general-purpose data processing and reporting:
- Ingests data from CSV, Excel, or JSON files with flexible validation
- Processes and analyzes any dataset (numeric and/or categorical data)
- Generates dynamic visualizations (bar chart, scatter plot) using Plotly
- Produces summary statistics and reports (PDF, Excel) with a download option
- Includes robust error handling and logging for reliability

Author: Jaswanth Mudapaka
Date: March 2025
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import json
import logging
import os
import matplotlib
matplotlib.use('Agg')  # Set thread-safe backend for matplotlib
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table
from reportlab.lib.styles import getSampleStyleSheet
from flask import Flask, render_template, request, send_file, url_for
import io
import shutil

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Temporary global variables for demo purposes (use session or file in production)
df_processed = None
freq_counts_df = None  # To store frequency counts as a DataFrame

def ingest_data(file, file_type='csv'):
    """
    Ingest data from an uploaded file (CSV, Excel, or JSON).
    """
    try:
        df = None
        if file_type == 'csv':
            encodings = ['utf-8', 'latin1', 'cp1252', 'utf-16']
            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding, sep=None, engine='python')
                    logging.info(f"Data ingested from uploaded file with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    logging.warning(f"Failed to read file with encoding: {encoding}")
                    continue
                except Exception as e:
                    logging.warning(f"Failed to parse file with encoding {encoding}: {e}")
                    continue
        elif file_type == 'excel':
            file.seek(0)
            df = pd.read_excel(file)
            logging.info("Data ingested from uploaded Excel file")
        elif file_type == 'json':
            file.seek(0)
            df = pd.read_json(file)
            logging.info("Data ingested from uploaded JSON file")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if df is None:
            raise ValueError("Unable to read the file with any supported encoding or format.")
        return df
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        return None

def validate_data(df):
    """
    Validate the dataset for basic integrity (non-empty).
    """
    try:
        if df.empty:
            return False, "Dataset is empty."
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                return False, f"Error converting column '{col}' to numeric: {str(e)}"
        
        return True, "Validation successful"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def preprocess_data(df):
    """
    Preprocess the dataset to handle missing values and normalize numeric data (if present).
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
        
        logging.info("Data preprocessed successfully")
        return df
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None

def generate_visualizations(df, filter_col=None, filter_value=None, output_dir="static/visualizations"):
    """
    Generate visualizations based on the dataset's structure.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        filtered_df = df.copy()
        if filter_col and filter_value:
            if filter_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[filter_col] == filter_value]
                logging.info(f"Filtered data by {filter_col}: {filter_value}")
            else:
                logging.warning(f"Filter column '{filter_col}' not found in dataset.")
        
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        categorical_cols = filtered_df.select_dtypes(include=['object']).columns
        
        visualization_paths = {}
        
        if len(numeric_cols) > 0:
            numeric_means = filtered_df[numeric_cols].mean()
            bar_fig = px.bar(x=numeric_means.values, y=numeric_means.index, 
                            orientation='h', title="Average Values of Numeric Columns")
            bar_path = os.path.join(output_dir, "mean_values_bar_chart.html")
            bar_fig.write_html(bar_path)
            visualization_paths['bar_chart'] = "visualizations/mean_values_bar_chart.html"
            logging.info(f"Mean values bar chart created at {bar_path}")
            
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                color_col = categorical_cols[0] if len(categorical_cols) > 0 else None
                scatter_fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col,
                                        hover_data=filtered_df.columns.tolist(),
                                        title=f"{x_col} vs. {y_col}")
                scatter_path = os.path.join(output_dir, "scatter_plot.html")
                scatter_fig.write_html(scatter_path)
                visualization_paths['scatter_plot'] = "visualizations/scatter_plot.html"
                logging.info(f"Scatter plot created at {scatter_path}: {x_col} vs. {y_col}")
            else:
                logging.info("Not enough numeric columns for scatter plot.")
        else:
            if len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                freq_counts = filtered_df[cat_col].value_counts()
                bar_fig = px.bar(x=freq_counts.index, y=freq_counts.values,
                                title=f"Frequency of {cat_col}")
                bar_fig.update_layout(xaxis_title=cat_col, yaxis_title="Count")
                bar_path = os.path.join(output_dir, "categorical_frequency_bar_chart.html")
                bar_fig.write_html(bar_path)
                visualization_paths['bar_chart'] = "visualizations/categorical_frequency_bar_chart.html"
                logging.info(f"Categorical frequency bar chart created at {bar_path} for {cat_col}")
            else:
                logging.warning("No numeric or categorical columns found for visualization.")
                return False, {}
        
        return True, visualization_paths
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        return False, {}

def generate_report(df, summary=None, output_dir="static/reports"):
    """
    Generate PDF and Excel reports for the dataset.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_file = os.path.join(output_dir, "data_report.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = [Paragraph("Data Analysis Report", styles['Title'])]
        
        if summary is not None:
            summary_table = Table(summary.round(2).to_numpy().tolist())
            elements.append(summary_table)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            freq_counts = df[cat_col].value_counts().reset_index()
            freq_counts.columns = [cat_col, 'Count']
            freq_table = Table(freq_counts.values.tolist(), colWidths=[100, 100])
            elements.append(Paragraph(f"\nFrequency of {cat_col}", styles['Heading2']))
            elements.append(freq_table)
        
        doc.build(elements)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols].mean().plot(kind='bar')
            plt.title("Average Values of Numeric Columns")
        elif len(categorical_cols) > 0:
            freq_counts = df[cat_col].value_counts()
            freq_counts.plot(kind='bar')
            plt.title(f"Frequency of {cat_col}")
        else:
            logging.warning("No data to plot in report.")
            return False, {}
        
        plt.savefig(os.path.join(output_dir, "chart.png"))
        plt.close()

        excel_file = os.path.join(output_dir, "data_report.xlsx")
        df.to_excel(excel_file, index=False)
        logging.info(f"Reports generated successfully: PDF at {pdf_file}, Excel at {excel_file}")
        return True, {'pdf_report': "reports/data_report.pdf", 'excel_report': "reports/data_report.xlsx"}
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        return False, {}

@app.route('/')
def index():
    """
    Render the main page with the upload form.
    """
    return render_template('index.html', results=False)

@app.route('/process', methods=['POST'])
def process():
    """
    Handle form submission, process the data, and render results.
    """
    global df_processed, freq_counts_df  # Temporary storage (use Flask session or file storage in production)

    # Get form data
    file = request.files['file']
    file_type = request.form['file_type']
    filter_col = request.form['filter_col']
    filter_value = request.form['filter_value']

    # Ingest data
    df = ingest_data(file, file_type)
    if df is None:
        return render_template('index.html', results=False, error="Error: Unable to read the file.")

    # Validate data
    is_valid, validation_message = validate_data(df)
    if not is_valid:
        return render_template('index.html', results=False, error=f"Error: {validation_message}")

    # Preprocess data
    df_processed = preprocess_data(df)
    if df_processed is None:
        return render_template('index.html', results=False, error="Error: Unable to preprocess the data.")

    # Generate summary statistics as a DataFrame
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    summary_df = None
    if len(numeric_cols) > 0:
        summary_df = df_processed[numeric_cols].describe().round(2).reset_index()
    else:
        summary_df = pd.DataFrame({"Message": ["No numeric data available."]})

    # Generate frequency counts for categorical columns as a DataFrame
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    freq_counts_df = pd.DataFrame()  # Reset the global DataFrame
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            counts = df_processed[col].value_counts().reset_index()
            counts.columns = [col, 'Count']
            freq_counts_df = counts  # Store for the new tab
    else:
        freq_counts_df = pd.DataFrame({"Message": ["No categorical data available."]})

    # Generate visualizations
    success, vis_paths = generate_visualizations(df_processed, filter_col, filter_value)
    if not success:
        logging.error("Visualization generation failed.")
        vis_paths = {}
    logging.info(f"Visualization paths: {vis_paths}")

    # Generate reports
    success, report_paths = generate_report(df_processed, df_processed[numeric_cols].describe() if len(numeric_cols) > 0 else None)
    if not success:
        logging.error("Report generation failed.")
        report_paths = {}
    logging.info(f"Report paths: {report_paths}")

    # Add links to view the table and frequency counts in new tabs
    table_link = url_for('view_table', _external=True)
    freq_counts_link = url_for('view_freq_counts', _external=True) if len(categorical_cols) > 0 else None

    # Render results with the table link
    return render_template('index.html', results=True, summary_df=summary_df, freq_counts_df=freq_counts_df,
                          bar_chart=vis_paths.get('bar_chart'), scatter_plot=vis_paths.get('scatter_plot'),
                          pdf_report=report_paths.get('pdf_report'), excel_report=report_paths.get('excel_report'),
                          table_link=table_link, freq_counts_link=freq_counts_link)

@app.route('/view_table')
def view_table():
    """
    Render the processed DataFrame as an HTML table in a new tab.
    """
    global df_processed
    if 'df_processed' not in globals() or df_processed is None:
        return "No data available to display.", 400
    
    # Convert DataFrame to HTML table with some basic styling
    table_html = df_processed.to_html(classes='table table-striped', index=False, border=0)
    
    # Simple HTML template for the new tab
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Processed Data Table</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            .table {{ width: 90%; margin: 20px auto; }}
            body {{ font-family: Arial, sans-serif; background: linear-gradient(135deg, #1e3c72, #2a5298); color: #fff; }}
            h2 {{ text-align: center; color: #fff; }}
        </style>
    </head>
    <body>
        <h2>Processed Data Table</h2>
        {table_html}
    </body>
    </html>
    """
    return html_content

@app.route('/view_freq_counts')
def view_freq_counts():
    """
    Render the frequency counts as an HTML table in a new tab.
    """
    global freq_counts_df
    if 'freq_counts_df' not in globals() or freq_counts_df.empty:
        return "No frequency counts available to display.", 400
    
    # Convert frequency counts DataFrame to HTML table with some basic styling
    table_html = freq_counts_df.to_html(classes='table table-striped', index=False, border=0)
    
    # Simple HTML template for the new tab
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Frequency Counts</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            .table {{ width: 90%; margin: 20px auto; }}
            body {{ font-family: Arial, sans-serif; background: linear-gradient(135deg, #1e3c72, #2a5298); color: #fff; }}
            h2 {{ text-align: center; color: #fff; }}
        </style>
    </head>
    <body>
        <h2>Frequency Counts for Categorical Data</h2>
        {table_html}
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    app.run(debug=True)