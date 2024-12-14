import os
import json
import asyncio
from fpdf import FPDF
import tempfile
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from data import data 

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")

tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_social_media_data",
            "description": "Analyze social media posts to count the distribution of users in specified regions (A, B, C, D) and identify posts not related to any specific region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "description": "A justification for why the content is associated with a specific region. This could include references to keywords, events, or locations mentioned in the post.",
                        "type": "string"
                    },
                    "region": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D", "Not related to any region"]
                    }
                },
                "required": ["region"]
            }
        }
    }
]

report_prompt = """You are an AI tasked with generating a comprehensive analysis report based on social media data. 
Your report should include:
1. An overview of the user distribution across specified regions (A, B, C, D) based on the provided data.
2. An analysis of the reasons for user distribution, including insights derived from the reasons collected during the analysis.
3. An explanation of why certain regions may have higher user counts based on the trends in the data.
4. Include the provided reasons for the distribution as supporting evidence.
5. Summarize the findings in a clear and concise manner.
"""

prompt = """You are an AI designed to analyze social media data and provide insights on user distribution across specified regions. Your tasks include:
1. Receiving a list of social media posts, each containing content, a user identifier, and an optional like count.
2. Analyzing each post to determine if it is related to one of the specified regions (A, B, C, D) or if it is not related to any region.
3. Counting the number of posts in each relevant region and returning a summary of the user distribution.
4. Providing clear and concise visualizations (e.g., bar charts) to represent the distribution of users across the regions.
Be efficient, precise, and ensure that your responses are structured for easy understanding. Always validate the input data format before processing, 
including checking for the presence of content, user, and region information."""

client = AsyncAzureOpenAI(
    azure_endpoint="https://hkust.azure-api.net",  # HKUST Azure end point
    api_key=api_key,
    api_version="2024-02-01",
)

region_count = {"A": 0, "B": 0, "C": 0, "D": 0, "Not related to any region": 0}
all_reasons = ""

async def process_posts(posts):
    global all_reasons  # Declare all_reasons as global to modify it
    for post in posts:
        content = post['post']['content']  # Accessing content from the post
        full_message = f"{prompt}\n\nPost content: {content}"
        
        # Create a chat completion for each post
        raw_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": full_message}
            ],
            temperature=0.5,
            tools=tools,
            tool_choice={"type": "function", "function": {
                "name": "analyze_social_media_data"}},
        )
        
        arguments_str = raw_response.choices[0].message.tool_calls[0].function.arguments
        arguments = json.loads(arguments_str)
        
        region = arguments.get("region")
        reason = arguments.get("reason", "")  # Default to empty string if not found
        
        print(f"Region: {region}, Reason: {reason}")

        if region in region_count:
            region_count[region] += 1
        
        if reason:
            all_reasons += f"{reason} "

async def generate_report(all_reasons, data):
    # Create the report content
    report_message = f"{report_prompt}\n\nReasons for Distribution:\n{all_reasons}\n\nData:\n{json.dumps(data, indent=2)}"

    # Call the AI to generate the report
    report_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": report_message}
        ],
        temperature=0.5
    )

    # Extract the report text
    report_text = report_response.choices[0].message.content
    return report_text

def generate_chart_image(region_count):
    # Create a bar chart
    regions = list(region_count.keys())
    counts = list(region_count.values())

    plt.figure(figsize=(10, 6))  # Set figure size
    plt.bar(regions, counts, color=['blue', 'orange', 'green', 'red','purple'])
    plt.xlabel('Region')
    plt.ylabel('Number of Users')
    plt.title('User Distribution Across Regions')
    plt.xticks(rotation=0)
    plt.grid(axis='y')

    # Save the plot to a BytesIO object
    img_bytes = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_bytes, format='png')
    plt.close()  # Close the plot to free memory
    img_bytes.seek(0)  # Go to the start of the BytesIO object
    return img_bytes

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Social Media Analysis Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(report_text, chart_image_bytes):
    pdf = PDF()
    
    # Add a TrueType font
    pdf.add_font("Arial", "", "Arial.ttf", uni=True)  # Adjust the path if necessary
    pdf.set_font("Arial", size=12)
    
    pdf.add_page()

    

    # Use tempfile to create a temporary file for the image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_file:
        img_file.write(chart_image_bytes.getvalue())
        img_file_path = img_file.name

    # Add the chart image
    pdf.image(img_file_path, x=10, y=pdf.get_y() + 10, w=190)  # Adjust the position and size as needed
    
    # Adjust the y position for the text to avoid overlap
    pdf.set_y(pdf.get_y() + 150)  # Move down by 100 units (adjust based on image height)

    # Add the report text
    pdf.multi_cell(0, 10, report_text)
    
    # Save the PDF
    pdf_file_path = "social_media_analysis_report.pdf"
    pdf.output(pdf_file_path)
    print(f"Report saved as {pdf_file_path}")

    # Optionally, delete the temporary image file after use
    os.remove(img_file_path)
async def main(data):
    await process_posts(data)  # Process the posts first
    report_text = await generate_report(all_reasons, data)  # Generate the report
    chart_image_bytes = generate_chart_image(region_count)  # Generate the chart image
    create_pdf(report_text, chart_image_bytes)  # Create the PDF with the report and chart

# Run the main function
asyncio.run(main(data))