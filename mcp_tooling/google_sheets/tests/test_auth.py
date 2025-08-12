"""Unit tests for GoogleSheetsAuth module"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from mcp_tooling.google_sheets.api.auth import GoogleSheetsAuth

class TestGoogleSheetsAuth:
    """Test cases for GoogleSheetsAuth functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth = GoogleSheetsAuth()
    
    def test_scope_initialization(self):
        """Test scope initialization"""
        # Test default scope
        auth = GoogleSheetsAuth()
        expected_full_scopes = ['https://www.googleapis.com/auth/spreadsheets']
        assert auth.scopes == expected_full_scopes
        
        # Test readonly scope
        auth = GoogleSheetsAuth(scope_level='readonly')
        expected_readonly_scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        assert auth.scopes == expected_readonly_scopes
        
        # Test drive scope
        auth = GoogleSheetsAuth(scope_level='drive')
        expected_drive_scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        assert auth.scopes == expected_drive_scopes
    
    @patch('os.path.exists')
    @patch('mcp_tooling.google_sheets.api.auth.service_account')
    @patch('mcp_tooling.google_sheets.api.auth.build')
    def test_authenticate_service_account_success(self, mock_build, mock_service_account, mock_exists):
        """Test successful service account authentication"""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock credentials and service
        mock_creds = Mock()
        mock_creds.service_account_email = 'test@serviceaccount.com'
        mock_service_account.Credentials.from_service_account_file.return_value = mock_creds
        
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Test authentication
        result = self.auth.authenticate_service_account('test_key.json')
        
        # Verify calls
        mock_service_account.Credentials.from_service_account_file.assert_called_once_with(
            'test_key.json',
            scopes=self.auth.scopes
        )
        mock_build.assert_called_once_with('sheets', 'v4', credentials=mock_creds)
        
        # Verify state
        assert self.auth.creds == mock_creds
        assert self.auth.service == mock_service
        assert self.auth.auth_method == 'service_account'
        assert result == mock_service
    
    @patch('os.path.exists')
    def test_authenticate_service_account_file_not_found(self, mock_exists):
        """Test service account authentication with missing file"""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            self.auth.authenticate_service_account('nonexistent.json')
    
    @patch('os.environ.get')
    @patch('os.path.exists')
    @patch('mcp_tooling.google_sheets.api.auth.service_account')
    @patch('mcp_tooling.google_sheets.api.auth.build')
    def test_authenticate_with_google_application_credentials(
        self, mock_build, mock_service_account, mock_exists, mock_env_get
    ):
        """Test authentication using GOOGLE_APPLICATION_CREDENTIALS"""
        # Mock environment variable
        mock_env_get.return_value = '/path/to/service_account.json'
        mock_exists.return_value = True
        
        # Mock credentials and service
        mock_creds = Mock()
        mock_creds.service_account_email = 'test@serviceaccount.com'
        mock_service_account.Credentials.from_service_account_file.return_value = mock_creds
        
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Test authentication
        result = self.auth.authenticate()
        
        # Verify environment variable was checked
        mock_env_get.assert_called_with('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Verify service account authentication was called
        mock_service_account.Credentials.from_service_account_file.assert_called_once_with(
            '/path/to/service_account.json',
            scopes=self.auth.scopes
        )
        
        assert result == mock_service
        assert self.auth.auth_method == 'service_account'
    
    @patch('os.environ.get')
    @patch('os.path.exists')
    @patch('mcp_tooling.google_sheets.api.auth.google.auth')
    @patch('mcp_tooling.google_sheets.api.auth.build')
    def test_authenticate_with_adc(self, mock_build, mock_google_auth, mock_exists, mock_env_get):
        """Test authentication using Application Default Credentials"""
        # Mock no GOOGLE_APPLICATION_CREDENTIALS
        mock_env_get.return_value = None
        mock_exists.return_value = False
        
        # Mock ADC
        mock_creds = Mock()
        mock_project = 'test-project'
        mock_google_auth.default.return_value = (mock_creds, mock_project)
        
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        # Test authentication
        result = self.auth.authenticate(use_adc=True)
        
        # Verify ADC was called
        mock_google_auth.default.assert_called_once_with(scopes=self.auth.scopes)
        mock_build.assert_called_once_with('sheets', 'v4', credentials=mock_creds)
        
        assert result == mock_service
        assert self.auth.auth_method == 'adc'
    
    @patch('os.environ.get')
    @patch('os.path.exists')
    def test_authenticate_no_credentials_available(self, mock_exists, mock_env_get):
        """Test authentication failure when no credentials are available"""
        # Mock no environment variable and no files exist
        mock_env_get.return_value = None
        mock_exists.return_value = False
        
        with pytest.raises(ValueError) as exc_info:
            self.auth.authenticate(use_adc=False)
        
        assert "No authentication method available" in str(exc_info.value)
        assert "GOOGLE_APPLICATION_CREDENTIALS" in str(exc_info.value)
    
    def test_get_service_not_authenticated(self):
        """Test get_service when not authenticated"""
        with pytest.raises(ValueError) as exc_info:
            self.auth.get_service()
        
        assert "Not authenticated" in str(exc_info.value)
    
    def test_get_auth_info_not_authenticated(self):
        """Test get_auth_info when not authenticated"""
        result = self.auth.get_auth_info()
        assert result == {"authenticated": False}
    
    def test_get_auth_info_service_account(self):
        """Test get_auth_info for service account authentication"""
        # Mock authenticated state
        mock_creds = Mock()
        mock_creds.service_account_email = 'test@serviceaccount.com'
        
        self.auth.creds = mock_creds
        self.auth.auth_method = 'service_account'
        
        result = self.auth.get_auth_info()
        
        expected = {
            "authenticated": True,
            "method": "service_account",
            "scopes": self.auth.scopes,
            "service_account_email": "test@serviceaccount.com"
        }
        
        assert result == expected
    
    def test_get_auth_info_oauth(self):
        """Test get_auth_info for OAuth authentication"""
        # Mock authenticated state
        mock_creds = Mock()
        # OAuth credentials don't have service_account_email
        if hasattr(mock_creds, 'service_account_email'):
            delattr(mock_creds, 'service_account_email')
        
        self.auth.creds = mock_creds
        self.auth.auth_method = 'oauth'
        
        result = self.auth.get_auth_info()
        
        expected = {
            "authenticated": True,
            "method": "oauth",
            "scopes": self.auth.scopes
        }
        
        assert result == expected
        # Should not include service_account_email for OAuth
        assert "service_account_email" not in result