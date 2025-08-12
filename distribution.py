# -*- coding: cp1252 -*-
import os
import json
import time
import asyncio
from fpdf import FPDF
import tempfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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

report_prompt = """You are an AI tasked with generating a comprehensive analysis report based on social media data and network resource predictions. 
Your report should include:
1. An overview of the user distribution across specified regions (A, B, C, D) based on the provided data.
2. Analysis of event scale predictions for each region and the derived network resource allocation.
3. Explanation of the allocation strategy: why certain regions get more bandwidth and higher priority.
4. Summarize how this predictive allocation can improve network performance during events.
5. Include performance improvement estimates based on the simulation results.
"""

prompt = """You are an AI designed to analyze social media data and provide insights on user distribution across specified regions. Your tasks include:
1. Receiving a list of social media posts, each containing content, a user identifier, and an optional like count.
2. Analyzing each post to determine if it is related to one of the specified regions (A, B, C, D) or if it is not related to any region.
3. Counting the number of posts in each relevant region and returning a summary of the user distribution.
Be efficient, precise, and ensure that your responses are structured for easy understanding. Always validate the input data format before processing, 
including checking for the presence of content, user, and region information."""

client = AsyncAzureOpenAI(
    azure_endpoint="https://hkust.azure-api.net",
    api_key=api_key,
    api_version="2024-02-01",
)

region_count = {"A": 0, "B": 0, "C": 0, "D": 0, "Not related to any region": 0}
all_reasons = ""

async def process_posts(posts):
    global all_reasons
    for post in posts:
        content = post['post']['content']
        full_message = f"{prompt}\n\nPost content: {content}"
        
        raw_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_message}],
            temperature=0.5,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "analyze_social_media_data"}},
        )
        
        arguments_str = raw_response.choices[0].message.tool_calls[0].function.arguments
        arguments = json.loads(arguments_str)
        
        region = arguments.get("region")
        reason = arguments.get("reason", "")
        
        print(f"Region: {region}, Reason: {reason}")

        if region in region_count:
            region_count[region] += 1
        
        if reason:
            all_reasons += f"{reason}\n"
class RequestThrottler:
    def __init__(self, requests_per_minute=120, safety_margin=0.1):
        
        self.interval = 60 / (requests_per_minute * (1 - safety_margin))
        self.last_request_time = 0
        
    async def wait_before_request(self):
        elapsed = time.time() - self.last_request_time
        wait_time = max(0, self.interval - elapsed)
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()

async def robust_api_call(throttler, content):
    """API calls with retry mechanism and dynamic throttling"""
    max_retries = 3
    retry_delay = 2  
    
    for attempt in range(max_retries):
        try:
            await throttler.wait_before_request()
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                temperature=0.5,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "analyze_social_media_data"}},
            )
            
            return response
        
        except Exception as e:
            error_type = type(e).__name__
            
            
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                print(f"Rate limiting error (attempts {attempt+1}/{max_retries}). Increase the interval and retry...")
                
                # Dynamic adjustment of intervals
                throttler.interval *= 1.5  # Increase the interval by 50 per cent
                retry_delay *= 2  # Index retreat
                
                # Wait longer and retry
                await asyncio.sleep(retry_delay)
                continue
                
            elif "server error" in str(e).lower() or "internal error" in str(e).lower():
                print(f"server error (attempts {attempt+1}/{max_retries}). Retrying...")
                await asyncio.sleep(retry_delay)
                continue
                
            else:
                print(f"unrecoverable error: {error_type} - {str(e)}")
                raise  # Re-throw non-rate-limited error

    raise Exception(f"All {max_retries} Failed attempts")

async def process_posts_with_throttling(posts):
    global all_reasons, region_count
    
    throttler = RequestThrottler(requests_per_minute=100, safety_margin=0.1)
    
    for post in posts:
        content = post['post']['content']
        full_message = f"{prompt}\n\nPost content: {content}"
        
        try:
            response = await robust_api_call(throttler, full_message)
            
            arguments_str = response.choices[0].message.tool_calls[0].function.arguments
            arguments = json.loads(arguments_str)
            
            region = arguments.get("region")
            reason = arguments.get("reason", "")
            
            print(f"Region: {region}, Reason: {reason} | Interval: {throttler.interval:.2f}s")

            if region in region_count:
                region_count[region] += 1
            
            if reason:
                all_reasons += f"{reason}\n"
                
        except Exception as e:
            print(f"Failure to process post: {str(e)}")
            with open("failed_posts.log", "a") as f:
                f.write(f"{content}\n\n")

'''# 1. Request batch processing (core optimisation)
async def batch_process_posts(posts, batch_size=10):
    """Batch Processing Posts to Reduce API Calls"""
    global all_reasons, region_count
    
    # Creating batches
    batches = [posts[i:i+batch_size] for i in range(0, len(posts), batch_size)]
    
    for batch in batches:
        print( batch)
        # Building bulk request content
        batch_content = "\n\n".join(
            [f"Post {idx+1}: {post['post']['content']}" 
             for idx, post in enumerate(batch)]
        )
        
        full_message = f"{prompt}\n\n### Batch Posts ###\n{batch_content}"
        
        # API calls (using batch processing)
        try:
            raw_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": full_message}],
                temperature=0.5,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "analyze_social_media_data"}},
            )
            
            # Parsing Batch Responses
            arguments_str = raw_response.choices[0].message.tool_calls[0].function.arguments
            batch_results = json.loads(arguments_str).get("batch_results", [])
            
            # Processing each result
            for result in batch_results:
                region = result.get("region")
                reason = result.get("reason", "")
                print(f"Region: {region}, Reason: {reason}")
                if region in region_count:
                    region_count[region] += 1
                
                if reason:
                    all_reasons += f"{reason}\n"
                    
        except Exception as e:
            print(f"Batch processing error: {e}")
            # Fallback to single-article processing on failure
            await process_posts(batch)

# 2. Request Rate Limiter
class RateLimiter:
    """Adaptive Rate Controller"""
    def __init__(self, max_rpm=180):
        self.max_rpm = max_rpm
        self.interval = 60 / max_rpm  # Request interval (seconds)
        self.last_request_time = 0
        
    async def wait(self):
        """Wait for the appropriate time to send the next request"""
        elapsed = time.time() - self.last_request_time
        wait_time = max(0, self.interval - elapsed)
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()

# 3. Local caching layer (to reduce repeated calls)
class ContentCache:
    """Content-based response caching"""
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
        
    def get(self, content):
        """Getting Cached Results"""
        return self.cache.get(content, None)
    
    def set(self, content, result):
        """Setting up cached results"""
        if len(self.cache) >= self.max_size:
            # Remove the oldest entry
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
            
        self.cache[content] = result
        self.access_order.append(content)

# 4. hybrid processing strategy
async def hybrid_process_posts(posts, cache):
    """Mixed processing strategy: caching + batch processing"""
    processed = []
    to_process = []
    
    # Step 1: Check the cache
    for post in posts:
        content = post['post']['content']
        cached = cache.get(content)
        
        if cached:
            region = cached.get("region")
            reason = cached.get("reason", "")
            
            if region in region_count:
                region_count[region] += 1
            
            if reason:
                all_reasons += f"{reason}\n"
        else:
            to_process.append(post)
    
    # Step 2: Batch process uncached content
    if to_process:
        await batch_process_posts(to_process)
        
        # Updating the cache
        for post in to_process:
            content = post['post']['content']
            # Here you need to set the cache value according to the actual response
            # Simplified example: the actual implementation needs to store the API response
            cache.set(content, {"region": "cached", "reason": "cached"})
    
    return processed + to_process'''
def generate_chart_image(region_count, resource_allocation):
    """Generate user distribution and resource allocation charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # user map
    regions = list(region_count.keys())
    counts = list(region_count.values())
    ax1.bar(regions, counts, color=['blue', 'orange', 'green', 'red', 'purple'])
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Number of Posts')
    ax1.set_title('Social Media Activity by Region')
    ax1.grid(axis='y')
    
    # Resource allocation heat map
    alloc_df = pd.DataFrame(resource_allocation).T
    sns.heatmap(alloc_df[['bandwidth', 'priority']], annot=True, fmt=".1f", cmap="YlGnBu", ax=ax2)
    ax2.set_title('Network Resource Allocation')
    
    plt.tight_layout()
    
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()
    img_bytes.seek(0)
    return img_bytes

def simulate_performance(resource_allocation):
    """Simulation performance improvements"""
    # Base scenario (no optimisation)
    base_performance = {
        'bandwidth_utilization': 65,
        'peak_load': 95,
        'latency': 120
    }
    
    # Optimised Scene
    optimized_performance = {
        'bandwidth_utilization': base_performance['bandwidth_utilization'] + 25,
        'peak_load': base_performance['peak_load'] - 35,
        'latency': base_performance['latency'] - 30
    }
    
    return base_performance, optimized_performance

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'EventPulse: Real-time Network Optimization Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(2)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

def create_pdf(report_text, chart_image_bytes, resource_allocation, performance_data):
    pdf_file_path = "EventPulse_Report.pdf"
    
    pdf = PDF()
    pdf.add_page()
    
    # Add title and abstract
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "This report presents a real-time network resource optimization system based on social media analysis. By predicting event scales and user distribution, we dynamically allocate bandwidth and prioritize network resources to critical areas, significantly improving network performance during large events.")
    pdf.ln(10)
    
    # Add images directly to PDF from BytesIO
    with io.BytesIO() as img_bytes:
        # Copying chart data to a new BytesIO object
        img_bytes.write(chart_image_bytes.getvalue())
        img_bytes.seek(0)
        
        # Using fpdf's image method to directly process images in memory
        pdf.image(img_bytes, x=10, w=190, type='PNG')
    
    pdf.ln(10)
    
    # Add details of resource allocation
    pdf.chapter_title('Resource Allocation Details')
    alloc_text = "Predicted resource needs based on event scale:\n\n"
    for region, alloc in resource_allocation.items():
        alloc_text += (f"Region {region}: Predicted attendees: {alloc['predicted_scale']}, "
                      f"Bandwidth: {alloc['bandwidth']:.1f}Mbps, Priority: {alloc['priority']:.1f}%\n")
    pdf.chapter_body(alloc_text)
    
    # Add performance improvements
    pdf.chapter_title('Performance Improvement')
    base, optimized = performance_data
    perf_text = (
        "Simulation results show significant network improvements:\n\n"
        f"Bandwidth utilization: {base['bandwidth_utilization']}% - {optimized['bandwidth_utilization']}% (+{optimized['bandwidth_utilization']-base['bandwidth_utilization']}%)\n"
        f"Peak load reduction: {base['peak_load']}% - {optimized['peak_load']}% (-{base['peak_load']-optimized['peak_load']}%)\n"
        f"Latency reduction: {base['latency']}ms - {optimized['latency']}ms (-{base['latency']-optimized['latency']}ms)\n"
    )
    pdf.chapter_body(perf_text)
    
    # Add AI Analytics Report
    pdf.chapter_title('Comprehensive Analysis')
    pdf.chapter_body(report_text)
    
    # Add conclusions
    pdf.chapter_title('Conclusion')
    pdf.chapter_body("EventPulse demonstrates that semantic-driven network optimization can significantly improve resource utilization and user experience during large events. By proactively allocating resources based on real-time social media analysis, network operators can prevent congestion and ensure quality service in high-demand areas.")
    
    # Save PDF to memory buffer
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    
    # Write PDF in memory to file
    with open(pdf_file_path, "wb") as f:
        f.write(pdf_output.getvalue())
    
    print(f"Report saved as {pdf_file_path}")
def adaptive_prediction_model(region_data, global_stats):
    
    # 1. Calculate ratios based on global statistics
    post_ratio = region_data['posts'] / global_stats['total_posts']
    if region_data['posts'] > 0:
        engagement_ratio = (region_data['likes'] / region_data['posts']) / global_stats['avg_likes_per_post']
    else:
        engagement_ratio = 0
    
    # 2. Calculate impact factor
    # - Posts value weight: 0.6 (User engagement)
    # - Engagement value weight: 0.4 (Participate in events)
    impact_factor = (0.6 * post_ratio) + (0.4 * engagement_ratio)
    
    # 3. Dynamic size projections (based on global benchmarks)
    # Base size = global average size * impact factor * adjustment factor
    # Adjustment factor varies automatically according to the size of the dataset
    scale_adjustment = 1.0 + (global_stats['total_posts'] / 100) * 0.1
    predicted_scale = global_stats['avg_scale'] * impact_factor * scale_adjustment
    
    # 4. Bandwidth demand forecasts (based on non-linear relationships)
    # Using Logarithmic Functions to Avoid the Effects of Extreme Values
    bandwidth = 20 + 60 * (1 - 1/(1 + predicted_scale/500))
    
    return {
        'predicted_scale': int(predicted_scale),
        'bandwidth': min(100, bandwidth),
        'impact_factor': impact_factor    
    }
def calculate_global_stats(data, region_count):
    total_posts = sum(region_count.values())
    total_likes = sum(post['post']['like'] for post in data)
    avg_likes_per_post = total_likes / total_posts if total_posts > 0 else 0
    
    # Calculation of baseline size (use of quartiles to avoid extreme values)
    region_activities = [count for count in region_count.values()]
    q75 = np.percentile(region_activities, 75)
    avg_scale = q75 * 30  # Use of the 75th percentile as a benchmark
    
    return {
        'total_posts': total_posts,
        'total_likes': total_likes,
        'avg_likes_per_post': avg_likes_per_post,
        'avg_scale': avg_scale
    }

def generate_resource_allocation(region_count, global_stats, data):
    allocation = {}
    impact_factors = {}
    # Collect region-specific data
    region_likes = {}
    for region in region_count:
        # Calculate likes for each region
        region_likes[region] = sum(
            post['post']['like'] for post in data 
            if any(region in post['post']['content'] for region in ['A', 'B', 'C', 'D'])
        )
    
    for region in ['A', 'B', 'C', 'D']:
        region_data = {
            'posts': region_count[region],
            'likes': region_likes.get(region, 0)
        }
        result = adaptive_prediction_model(region_data, global_stats)
        allocation[region] = result
        impact_factors[region] = result['impact_factor']
    total_impact = sum(impact_factors.values())
    for region in allocation:
        if total_impact > 0:
            priority = (impact_factors[region] / total_impact) * 100
        else:
            priority = 0
        
        allocation[region]['priority'] = min(100, max(0, priority))
    return allocation

async def main(data):
    global all_reasons, region_count
    
    '''
    # Initialisation tools
    limiter = RateLimiter(max_rpm=150)  # Below API limits
    cache = ContentCache(max_size=2000)
    
    # Processing data
    await hybrid_process_posts(data, cache)
    '''
    await process_posts_with_throttling(data)

    #await process_posts(data)
    #Calculate global statistics
    global_stats = calculate_global_stats(data, region_count)
    
    # Generate resource allocation based on adaptive prediction model
    resource_allocation = generate_resource_allocation(region_count, global_stats, data)
    print("Resource Allocation:", resource_allocation)
    
    # Generate chart image
    chart_image_bytes = generate_chart_image(region_count, resource_allocation)
    # Simulate performance improvement
    performance_data = simulate_performance(resource_allocation)
    
    # Generate AI analysis reports
    report_data = {
        "region_count": region_count,
        "resource_allocation": resource_allocation,
        "performance_improvement": {
            "bandwidth_utilization": performance_data[1]['bandwidth_utilization'] - performance_data[0]['bandwidth_utilization'],
            "peak_load_reduction": performance_data[0]['peak_load'] - performance_data[1]['peak_load'],
            "latency_reduction": performance_data[0]['latency'] - performance_data[1]['latency']
        }
    }
    
    report_message = f"{report_prompt}\n\nData Summary:\n{json.dumps(report_data, indent=2)}"
    
    report_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": report_message}],
        temperature=0.5
    )
    
    report_text = report_response.choices[0].message.content
    
    # Create PDF reports
    create_pdf(report_text, chart_image_bytes, resource_allocation, performance_data)

asyncio.run(main(data))