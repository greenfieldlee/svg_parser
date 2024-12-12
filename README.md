## Prerequisites

- Python 3.8 or higher
- Bybit account (testnet for testing, live for real trading)
- Basic understanding of cryptocurrency trading and associated risks

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/greenfieldlee/svg_parser
   cd svg_parser
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration
   pm2 start main.py --name "fastapi-app" --interpreter /usr/bin/python3
