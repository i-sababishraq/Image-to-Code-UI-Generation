
from PIL import Image, ImageDraw

# WARNING: This is generated code and could be unsafe.
# Always run in a sandboxed environment.

try:
    img_path = r"C:\\Users\\mdieh\\Downloads\\test_image_1.png"
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    # These are hardcoded coordinates; a real agent would make them dynamic
    eye_center = (200, 250)
    radius = 30
    
    # Draw monocle lens
    draw.ellipse(
        (eye_center[0]-radius, eye_center[1]-radius, eye_center[0]+radius, eye_center[1]+radius), 
        outline='black', 
        width=5
    )
    
    # Draw chain
    draw.line(
        (eye_center[0]+radius, eye_center[1], eye_center[0]+radius+50, eye_center[1]+50),
        fill='black',
        width=3
    )
    
    img.save('output.png')
    print("Image with monocle saved to output.png")

except FileNotFoundError:
    print(f"Error: The file at {{img_path}} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
