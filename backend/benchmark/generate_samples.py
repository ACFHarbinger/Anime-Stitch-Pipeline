import os
import math
from PIL import Image, ImageDraw

def create_gradient_sample(output_dir, num_frames=6, width=400, height=600):
    os.makedirs(output_dir, exist_ok=True)
    
    # We create a tall canvas with a vertical gradient and crop a moving window
    canvas_height = height + num_frames * 50
    canvas = Image.new('RGB', (width, canvas_height))
    draw = ImageDraw.Draw(canvas)
    
    for y in range(canvas_height):
        r = int(255 * (y / canvas_height))
        g = int(100 + 100 * math.sin(y / 100))
        b = int(255 * (1 - y / canvas_height))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
        
    for i in range(num_frames):
        # Shift down by 50px each frame
        y_offset = i * 50
        frame = canvas.crop((0, y_offset, width, y_offset + height))
        
        # Add frame number text just to be helpful
        fdraw = ImageDraw.Draw(frame)
        fdraw.text((10, 10), f"Frame {i+1}", fill=(255, 255, 255))
        
        frame.save(os.path.join(output_dir, f"frame_{i:03d}.png"))

def create_pattern_sample(output_dir, num_frames=6, width=400, height=600):
    os.makedirs(output_dir, exist_ok=True)
    
    canvas_height = height + num_frames * 50
    canvas = Image.new('RGB', (width, canvas_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Draw a grid and dots
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, canvas_height)], fill='gray')
    for y in range(0, canvas_height, 40):
        draw.line([(0, y), (width, y)], fill='gray')
        
    for x in range(20, width, 40):
        for y in range(20, canvas_height, 40):
            draw.ellipse([x-10, y-10, x+10, y+10], fill='blue')
            
    for i in range(num_frames):
        y_offset = i * 50
        frame = canvas.crop((0, y_offset, width, y_offset + height))
        frame.save(os.path.join(output_dir, f"frame_{i:03d}.png"))

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/samples'))
    
    print("Generating sample A: test_scroll_gradient...")
    create_gradient_sample(os.path.join(base_dir, 'test_scroll_gradient'), num_frames=6)
    
    print("Generating sample B: test_scroll_pattern...")
    create_pattern_sample(os.path.join(base_dir, 'test_scroll_pattern'), num_frames=6)
    
    print("Samples generated successfully.")

if __name__ == '__main__':
    main()
