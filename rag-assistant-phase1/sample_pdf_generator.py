"""
Sample PDF content generator for testing RAG pipeline
"""

import os
from pathlib import Path

def get_sample_content():
    """Get sample content for PDF creation."""
    content = """SMART HOME LED CONTROLLER
MODEL: SH-LED-2024
USER MANUAL AND WARRANTY GUIDE

1. INTRODUCTION

Thank you for purchasing the Smart Home LED Controller SH-LED-2024. This device allows you to control up to 16 LED strips with advanced features including color mixing, dimming, scheduling, and voice control integration.

2. PACKAGE CONTENTS

- Smart Home LED Controller unit
- Power adapter (12V, 3A)  
- User manual
- Quick start guide
- Mounting hardware
- Remote control

3. TECHNICAL SPECIFICATIONS

- Input Voltage: 100-240V AC, 50/60Hz
- Output: 12V DC, 3A maximum
- LED Strip Support: Up to 16 channels
- Wireless: WiFi 802.11 b/g/n, Bluetooth 5.0
- Operating Temperature: -10°C to 50°C
- Dimensions: 150 x 100 x 25mm
- Weight: 250g

4. INSTALLATION GUIDE

4.1 Mounting the Controller
Mount the controller in a dry, well-ventilated area away from direct sunlight and heat sources. Use the provided mounting screws to secure the unit to a wall or surface.

4.2 Connecting LED Strips
Connect your LED strips to the output terminals. Each channel supports up to 2A current draw. Ensure proper polarity when connecting (red = positive, black = negative).

4.3 Power Connection
Connect the provided power adapter to the controller and plug into a standard wall outlet. The power LED will illuminate green when properly connected.

5. LED STATUS INDICATORS

The controller features several LED indicators to show system status:

- Power LED (Green): Steady when powered on
- WiFi LED (Blue): Steady when connected to network, blinking when connecting  
- Status LED (Multi-color): Green blinking for normal operation, Red blinking for error condition or overload, Blue blinking for pairing mode active, Orange steady for firmware update in progress

6. TROUBLESHOOTING

6.1 LED Blinking Patterns

If you notice unusual LED blinking patterns, refer to this guide:

- Rapid red blinking (5 times per second): Overload condition detected. Reduce LED strip load or check connections.
- Slow red blinking (once per second): Temperature protection active. Ensure adequate ventilation.
- Alternating red/blue: Network connection error. Check WiFi settings.
- Purple blinking: Low voltage detected. Check power adapter connection.

6.2 Common Issues

Problem: Controller not responding to remote
Solution: Check battery in remote control. Re-pair remote if necessary.

Problem: LED strips flickering
Solution: Check all connections are secure. Verify power supply capacity.

Problem: Cannot connect to WiFi
Solution: Reset network settings by holding setup button for 10 seconds.

7. WARRANTY INFORMATION

7.1 Warranty Period
This product is covered by a 2-year limited warranty from the date of purchase.

7.2 Warranty Coverage
The warranty covers defects in materials and workmanship under normal use. It does not cover damage caused by misuse or abuse, exposure to moisture or extreme temperatures, unauthorized modifications, or normal wear and tear.

7.3 Warranty Claim Process
To make a warranty claim: 1) Contact customer support with proof of purchase, 2) Describe the issue and provide photos if requested, 3) Follow return instructions if replacement is authorized, 4) Retain original packaging for returns.

8. TECHNICAL SUPPORT

For technical support and additional resources:
- Website: www.smarthome-tech.com/support
- Email: support@smarthome-tech.com
- Phone: 1-800-SMART-LED
- Live chat available 24/7 on our website

9. SPECIFICATIONS SUMMARY

Model: SH-LED-2024
Power: 12V DC, 36W maximum
Channels: 16 independent LED channels
Control: WiFi, Bluetooth, Remote, Voice commands
Compatibility: Alexa, Google Assistant, Apple HomeKit
App: SmartHome Controller (iOS/Android)
Firmware: Updateable via app

10. SAFETY WARNINGS

- Do not expose to water or moisture
- Do not exceed maximum current ratings
- Use only provided power adapter
- Keep away from children and pets
- Do not disassemble unit
- Disconnect power before making connections

Thank you for choosing Smart Home LED Controller. For the latest firmware updates and additional features, please visit our website regularly."""
    return content

def create_sample_pdf():
    """Create a sample PDF document using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        
        # Create sample_data directory if it doesn't exist
        sample_dir = Path("sample_data")
        sample_dir.mkdir(exist_ok=True)
        
        # Define the output file path
        pdf_path = sample_dir / "sample_document.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Get content and split into paragraphs
        content = get_sample_content()
        paragraphs = content.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
                # Choose style based on content
                if para.strip().isupper() or para.strip().startswith('SMART HOME'):
                    style = styles['Title']
                elif para.strip().startswith(tuple('123456789')):
                    style = styles['Heading1']
                elif any(para.strip().startswith(f'{i}.{j}') for i in range(1, 11) for j in range(1, 10)):
                    style = styles['Heading2']
                else:
                    style = styles['Normal']
                
                story.append(Paragraph(para.strip(), style))
                story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ Sample PDF created successfully: {pdf_path}")
        print(f"   File size: {pdf_path.stat().st_size} bytes")
        return str(pdf_path)
        
    except ImportError:
        print("❌ reportlab not installed. Creating text file instead...")
        return create_text_file()
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        print("Creating text file as fallback...")
        return create_text_file()

def create_text_file():
    """Create a text file as fallback when PDF creation fails."""
    # Create sample_data directory if it doesn't exist
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create text file
    text_path = sample_dir / "sample_content.txt"
    content = get_sample_content()
    
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✅ Sample text file created: {text_path}")
    return str(text_path)

if __name__ == "__main__":
    print("🚀 Creating sample PDF for testing...")
    created_file = create_sample_pdf()
    print(f"✅ Sample file ready: {created_file}")
