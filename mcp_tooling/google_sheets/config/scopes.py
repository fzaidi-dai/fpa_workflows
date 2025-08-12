"""Google Sheets API scopes configuration"""

# Standard Google Sheets API scopes
SHEETS_READONLY = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SHEETS_FULL = ['https://www.googleapis.com/auth/spreadsheets']
DRIVE_FULL = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Scope descriptions for documentation
SCOPE_DESCRIPTIONS = {
    'https://www.googleapis.com/auth/spreadsheets.readonly': 
        'Read-only access to Google Sheets',
    'https://www.googleapis.com/auth/spreadsheets': 
        'Full read/write access to Google Sheets',
    'https://www.googleapis.com/auth/drive': 
        'Full access to Google Drive (includes file creation/deletion)',
}

# Recommended scopes for different use cases
RECOMMENDED_SCOPES = {
    'mcp_server_production': SHEETS_FULL,  # Most MCP servers need read/write
    'mcp_server_readonly': SHEETS_READONLY,  # For read-only analysis
    'mcp_server_with_drive': DRIVE_FULL,  # If creating new spreadsheets
    'development': SHEETS_FULL,  # Good for development/testing
}