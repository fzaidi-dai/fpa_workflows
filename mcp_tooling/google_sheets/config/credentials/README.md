# Google Sheets API Credentials

This directory is for storing Google Sheets API credential files.

## Required Credentials

### For Production (Recommended): Service Account
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project or create a new one
3. Enable the Google Sheets API
4. Go to **APIs & Services** → **Credentials**
5. Click **Create Credentials** → **Service Account**
6. Fill out the service account details
7. Click **Create and Continue**
8. Download the JSON key file
9. Place it in this directory as `service_account.json`
10. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to this file

### For Development: OAuth 2.0
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project
3. Go to **APIs & Services** → **Credentials**
4. Click **Create Credentials** → **OAuth 2.0 Client ID**
5. Configure OAuth consent screen if not already done
6. Select **Desktop Application**
7. Download the JSON file
8. Place it in this directory as `credentials.json`

## Environment Variables

Set one of these in your `.env` file:

```bash
# For service account (recommended)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json

# Or place the file in this directory and use relative path
GOOGLE_APPLICATION_CREDENTIALS=mcp_tooling/google_sheets/config/credentials/service_account.json
```

## Permissions

Make sure your service account has these permissions:
- Google Sheets API enabled
- For the spreadsheets you want to access:
  - Share the spreadsheet with the service account email address
  - Grant "Editor" permissions (or "Viewer" for read-only access)

## Security Notes

- **NEVER** commit credential files to version control
- Keep credential files secure and access-restricted
- Use service accounts for production environments
- Rotate credentials periodically
- Use the minimum required permissions