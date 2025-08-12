"""Authentication module for Google Sheets API - optimized for MCP server usage"""
import os
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Optional, List
import json
import logging

logger = logging.getLogger(__name__)

class GoogleSheetsAuth:
    """Authentication handler optimized for service account usage in MCP servers"""
    
    # Scopes as defined in Google's documentation
    SCOPES = {
        'readonly': ['https://www.googleapis.com/auth/spreadsheets.readonly'],
        'full': ['https://www.googleapis.com/auth/spreadsheets'],
        'drive': [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
    }
    
    def __init__(self, scope_level: str = 'full'):
        self.creds = None
        self.service = None
        self.scopes = self.SCOPES.get(scope_level, self.SCOPES['full'])
        self.auth_method = None
    
    def authenticate(self, 
                    service_account_file: Optional[str] = None,
                    oauth_credentials_file: Optional[str] = None,
                    use_adc: bool = True) -> build:
        """
        Unified authentication method with service account priority.
        
        Priority order:
        1. Explicit service account file path
        2. GOOGLE_APPLICATION_CREDENTIALS environment variable (Google standard)
        3. Application Default Credentials (works with GOOGLE_APPLICATION_CREDENTIALS)
        4. Default service account location
        5. OAuth 2.0 (for development/testing)
        
        Args:
            service_account_file: Path to service account JSON key file
            oauth_credentials_file: Path to OAuth client credentials
            use_adc: Use Application Default Credentials (recommended, works with GOOGLE_APPLICATION_CREDENTIALS)
        
        Returns:
            Authenticated Google Sheets service instance
        """
        # Try explicit service account file first
        if service_account_file and os.path.exists(service_account_file):
            logger.info("Using explicit service account file")
            return self.authenticate_service_account(service_account_file)
        
        # Try GOOGLE_APPLICATION_CREDENTIALS environment variable (Google standard)
        google_app_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if google_app_creds and os.path.exists(google_app_creds):
            logger.info(f"Using service account from GOOGLE_APPLICATION_CREDENTIALS: {google_app_creds}")
            return self.authenticate_service_account(google_app_creds)
        
        # Try Application Default Credentials (works with GOOGLE_APPLICATION_CREDENTIALS)
        if use_adc:
            logger.info("Using Application Default Credentials")
            try:
                return self.authenticate_default()
            except Exception as e:
                logger.warning(f"ADC authentication failed: {e}")
        
        # Try default service account location
        default_service_account = "mcp_tooling/google_sheets/config/credentials/service_account.json"
        if os.path.exists(default_service_account):
            logger.info("Using service account from default location")
            return self.authenticate_service_account(default_service_account)
        
        # Fall back to OAuth for development
        if oauth_credentials_file and os.path.exists(oauth_credentials_file):
            logger.warning("Using OAuth authentication - not recommended for production")
            return self.authenticate_oauth(oauth_credentials_file)
        
        # Try default OAuth location
        default_oauth = "mcp_tooling/google_sheets/config/credentials/credentials.json"
        if os.path.exists(default_oauth):
            logger.warning("Using OAuth from default location - not recommended for production")
            return self.authenticate_oauth(default_oauth)
        
        raise ValueError(
            "No authentication method available. Please ensure one of the following:\n"
            "1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable (recommended)\n"
            "2. Provide explicit service account key file path\n"
            "3. Place service account key at mcp_tooling/google_sheets/config/credentials/service_account.json\n"
            "4. Configure Application Default Credentials with 'gcloud auth application-default login'\n"
            "5. Provide OAuth credentials file (for development only)\n\n"
            f"Current GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}"
        )
    
    def authenticate_service_account(self, 
                                    key_file: str = "config/credentials/service_account.json") -> build:
        """
        Service account authentication - RECOMMENDED for MCP servers.
        
        Service accounts are ideal for server-to-server interactions:
        - No user interaction required
        - Can be granted specific permissions
        - Works in headless environments
        - Suitable for production use
        
        Args:
            key_file: Path to service account JSON key file
        
        Returns:
            Authenticated Google Sheets service instance
        """
        if not os.path.exists(key_file):
            raise FileNotFoundError(f"Service account key file not found: {key_file}")
        
        self.creds = service_account.Credentials.from_service_account_file(
            key_file,
            scopes=self.scopes
        )
        self.service = build("sheets", "v4", credentials=self.creds)
        self.auth_method = "service_account"
        logger.info(f"Authenticated with service account: {self.creds.service_account_email}")
        return self.service
    
    def authenticate_oauth(self, 
                          credentials_file: str = "config/credentials/credentials.json",
                          token_file: str = "config/token.json") -> build:
        """
        OAuth authentication - for development and user-specific access.
        
        Note: This requires user interaction and is not suitable for
        automated MCP server operations in production.
        
        Args:
            credentials_file: Path to OAuth client credentials
            token_file: Path to store/retrieve access tokens
        
        Returns:
            Authenticated Google Sheets service instance
        """
        if not os.path.exists(credentials_file):
            raise FileNotFoundError(f"OAuth credentials file not found: {credentials_file}")
        
        creds = None
        
        # Token file stores the user's access and refresh tokens
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, self.scopes)
        
        # If there are no (valid) credentials, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_file, self.scopes
                )
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(token_file, "w") as token:
                token.write(creds.to_json())
        
        self.creds = creds
        self.service = build("sheets", "v4", credentials=creds)
        self.auth_method = "oauth"
        logger.info("Authenticated with OAuth 2.0")
        return self.service
    
    def authenticate_default(self) -> build:
        """
        Use Google's Application Default Credentials (ADC).
        
        This works in Google Cloud environments or when credentials
        are configured via gcloud CLI.
        
        Returns:
            Authenticated Google Sheets service instance
        """
        import google.auth
        creds, project = google.auth.default(scopes=self.scopes)
        self.creds = creds
        self.service = build("sheets", "v4", credentials=creds)
        self.auth_method = "adc"
        logger.info(f"Authenticated with Application Default Credentials (project: {project})")
        return self.service
    
    def get_service(self):
        """Get authenticated service instance"""
        if not self.service:
            raise ValueError("Not authenticated. Call authenticate() or authenticate_* method first.")
        return self.service
    
    def get_auth_info(self) -> dict:
        """Get information about current authentication"""
        if not self.creds:
            return {"authenticated": False}
        
        info = {
            "authenticated": True,
            "method": self.auth_method,
            "scopes": self.scopes
        }
        
        if self.auth_method == "service_account" and hasattr(self.creds, 'service_account_email'):
            info["service_account_email"] = self.creds.service_account_email
        
        return info