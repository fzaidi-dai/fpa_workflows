"""Spreadsheet operations following Google's official Python examples"""
from googleapiclient.errors import HttpError
from typing import Dict, Any, List, Optional

class SpreadsheetOperations:
    """Core operations matching official documentation patterns"""
    
    def __init__(self, service):
        self.service = service
    
    def create(self, title: str) -> str:
        """Create spreadsheet - matches official example exactly"""
        try:
            spreadsheet = {"properties": {"title": title}}
            spreadsheet = (
                self.service.spreadsheets()
                .create(body=spreadsheet, fields="spreadsheetId")
                .execute()
            )
            print(f"Spreadsheet ID: {spreadsheet.get('spreadsheetId')}")
            return spreadsheet.get("spreadsheetId")
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def create_with_sheets(self, title: str, sheet_names: List[str]) -> Dict[str, Any]:
        """Create spreadsheet with multiple sheets"""
        try:
            body = {
                'properties': {'title': title},
                'sheets': []
            }
            
            for i, sheet_name in enumerate(sheet_names):
                body['sheets'].append({
                    'properties': {
                        'sheetId': i,
                        'title': sheet_name,
                        'gridProperties': {
                            'rowCount': 1000,
                            'columnCount': 26
                        }
                    }
                })
            
            spreadsheet = self.service.spreadsheets().create(body=body).execute()
            return {
                'spreadsheet_id': spreadsheet['spreadsheetId'],
                'spreadsheet_url': spreadsheet.get('spreadsheetUrl'),
                'sheets': [sheet['properties']['title'] 
                          for sheet in spreadsheet['sheets']]
            }
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def get_metadata(self, spreadsheet_id: str) -> Dict[str, Any]:
        """Get spreadsheet metadata"""
        try:
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=spreadsheet_id
            ).execute()
            return spreadsheet
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def add_sheet(self, spreadsheet_id: str, 
                 sheet_name: str,
                 rows: int = 1000,
                 columns: int = 26) -> Dict[str, Any]:
        """Add a new sheet to existing spreadsheet"""
        try:
            request = {
                'addSheet': {
                    'properties': {
                        'title': sheet_name,
                        'gridProperties': {
                            'rowCount': rows,
                            'columnCount': columns
                        }
                    }
                }
            }
            
            result = self.service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={'requests': [request]}
            ).execute()
            
            return result['replies'][0]['addSheet']['properties']
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def delete_sheet(self, spreadsheet_id: str, sheet_id: int) -> bool:
        """Delete a sheet from spreadsheet"""
        try:
            request = {
                'deleteSheet': {
                    'sheetId': sheet_id
                }
            }
            
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={'requests': [request]}
            ).execute()
            
            return True
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def duplicate_sheet(self, spreadsheet_id: str, 
                       source_sheet_id: int,
                       new_sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """Duplicate an existing sheet"""
        try:
            request = {
                'duplicateSheet': {
                    'sourceSheetId': source_sheet_id,
                    'insertSheetIndex': 0
                }
            }
            
            if new_sheet_name:
                request['duplicateSheet']['newSheetName'] = new_sheet_name
            
            result = self.service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={'requests': [request]}
            ).execute()
            
            return result['replies'][0]['duplicateSheet']['properties']
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def update_sheet_properties(self, spreadsheet_id: str,
                               sheet_id: int,
                               title: Optional[str] = None,
                               grid_properties: Optional[Dict] = None) -> bool:
        """Update sheet properties"""
        try:
            request = {
                'updateSheetProperties': {
                    'properties': {
                        'sheetId': sheet_id
                    },
                    'fields': ''
                }
            }
            
            fields = []
            if title:
                request['updateSheetProperties']['properties']['title'] = title
                fields.append('title')
            
            if grid_properties:
                request['updateSheetProperties']['properties']['gridProperties'] = grid_properties
                fields.append('gridProperties')
            
            request['updateSheetProperties']['fields'] = ','.join(fields)
            
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={'requests': [request]}
            ).execute()
            
            return True
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error