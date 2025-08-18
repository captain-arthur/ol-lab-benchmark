import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class GoogleSheetManager:
    """구글 시트 관리 클래스"""
    
    # 구글 시트 API 스코프
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    
    def __init__(self, spreadsheet_id: str, credentials_path: str = "secrets/client_secret.json", 
                 token_path: str = "secrets/token.json"):
        """
        GoogleSheetManager 초기화
        
        Args:
            spreadsheet_id: 구글 시트 ID
            credentials_path: 클라이언트 시크릿 파일 경로
            token_path: 토큰 파일 경로
        """
        self.spreadsheet_id = spreadsheet_id
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """구글 API 인증"""
        creds = None
        
        # 토큰 파일이 있으면 로드
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
        
        # 유효한 인증 정보가 없거나 만료된 경우
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # 토큰을 파일에 저장
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('sheets', 'v4', credentials=creds)
    
    def get_sheet_data(self, range_name: str) -> List[List[Any]]:
        """
        시트에서 데이터 가져오기
        
        Args:
            range_name: 범위 (예: 'Sheet1!A:Z')
            
        Returns:
            시트 데이터
        """
        try:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id, range=range_name
            ).execute()
            return result.get('values', [])
        except HttpError as error:
            print(f"시트 데이터 가져오기 오류: {error}")
            return []
    
    def append_data(self, range_name: str, values: List[List[Any]]) -> bool:
        """
        시트에 데이터 추가
        
        Args:
            range_name: 범위 (예: 'Sheet1!A:Z')
            values: 추가할 데이터
            
        Returns:
            성공 여부
        """
        try:
            body = {'values': values}
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            print(f"데이터 추가 완료: {result.get('updates').get('updatedRows')}행 추가됨")
            return True
        except HttpError as error:
            print(f"데이터 추가 오류: {error}")
            return False
    
    def check_data_exists(self, range_name: str, search_data: List[Any]) -> bool:
        """
        특정 데이터가 시트에 존재하는지 확인
        
        Args:
            range_name: 범위 (예: 'Sheet1!A:Z')
            search_data: 검색할 데이터 (첫 번째 행 기준)
            
        Returns:
            데이터 존재 여부
        """
        existing_data = self.get_sheet_data(range_name)
        
        if not existing_data:
            return False
        
        # 헤더 행 찾기 (query, passage, relevant가 포함된 행)
        header_row_idx = -1
        for i, row in enumerate(existing_data):
            if 'query' in row and 'passage' in row and 'relevant' in row:
                header_row_idx = i
                break
        
        if header_row_idx == -1:
            return False
        
        # 헤더 다음 행부터 검색
        for row in existing_data[header_row_idx + 1:]:
            if len(row) >= len(search_data):
                # 검색 데이터와 일치하는지 확인
                if all(str(row[i]) == str(search_data[i]) for i in range(len(search_data))):
                    return True
        
        return False
    
    def upload_data_with_timestamp(self, sheet_name: str, data: List[Dict[str, Any]], 
                                  key_columns: List[str] = None) -> bool:
        """
        타임스탬프와 함께 데이터 업로드 (중복 체크 포함)
        
        Args:
            sheet_name: 시트 이름
            data: 업로드할 데이터 리스트
            key_columns: 중복 체크할 키 컬럼들 (None이면 첫 번째 컬럼만 사용)
            
        Returns:
            성공 여부
        """
        if not data:
            print("업로드할 데이터가 없습니다.")
            return False
        
        range_name = f"{sheet_name}!A:Z"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        success_count = 0
        skip_count = 0
        
        # 중복 체크를 위한 키 데이터 생성
        if key_columns:
            check_data_list = [[str(item.get(key, "")) for key in key_columns] for item in data]
        else:
            check_data_list = [[str(list(item.values())[0])] for item in data]
        
        # 기존 데이터 확인
        existing_data = self.get_sheet_data(range_name)
        if existing_data and len(existing_data) > 2:
            print(f"기존 데이터가 존재합니다: {len(existing_data)}행")
            print("기존 데이터를 삭제하고 새로 업로드합니다.")
            
            # 기존 데이터 삭제
            try:
                self.service.spreadsheets().values().clear(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name
                ).execute()
                print("기존 데이터 삭제 완료")
            except Exception as e:
                print(f"데이터 삭제 중 오류 (무시): {e}")
        else:
            print("기존 데이터가 없습니다. 새로 업로드합니다.")
        
        # 업로드할 데이터 준비
        upload_rows = []
        
        # 첫 번째 행: 업로드 시간 (별도 행으로)
        time_row = [f"업로드 시간: {current_time}"]
        upload_rows.append(time_row)
        
        # 두 번째 행: 빈 행 (구분용)
        upload_rows.append([])
        
        # 세 번째 행: 컬럼명 (헤더)
        if data:
            header_row = list(data[0].keys())
            upload_rows.append(header_row)
        
        # 데이터 행들
        for item in data:
            row_data = []
            for key in data[0].keys():
                value = item.get(key, "")
                row_data.append(str(value) if value is not None else "")
            upload_rows.append(row_data)
        
        # 데이터 추가
        if self.append_data(range_name, upload_rows):
            success_count = len(data)
            print(f"데이터 업로드 성공: {len(data)}개 행 추가됨")
        else:
            print("데이터 업로드 실패")
            return False
        
        print(f"업로드 완료 - 성공: {success_count}개")
        return success_count > 0
    
    def get_sheet_info(self) -> Dict[str, Any]:
        """
        시트 정보 가져오기
        
        Returns:
            시트 정보
        """
        try:
            result = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()
            return result
        except HttpError as error:
            print(f"시트 정보 가져오기 오류: {error}")
            return {}


# 사용 예시
def main():
    """사용 예시"""
    # 시트 ID 설정
    SPREADSHEET_ID = "1yCmOp7__YWKB4k4pUWSF7DHfqvaR2-WVLnPkz5AfzN4"
    
    # GoogleSheetManager 인스턴스 생성
    sheet_manager = GoogleSheetManager(SPREADSHEET_ID)
    
    # 시트 정보 확인
    sheet_info = sheet_manager.get_sheet_info()
    print("시트 정보:", sheet_info.get('properties', {}).get('title', 'Unknown'))
    
    # 업로드할 샘플 데이터
    sample_data = [
        {"name": "홍길동", "age": "30", "city": "서울"},
        {"name": "김철수", "age": "25", "city": "부산"},
        {"name": "이영희", "age": "28", "city": "대구"},
    ]
    
    # 데이터 업로드 (중복 체크 포함)
    success = sheet_manager.upload_data_with_timestamp(
        sheet_name="Sheet1",  # 시트 이름을 실제 시트 이름으로 변경하세요
        data=sample_data,
        key_columns=["name"]  # name 컬럼으로 중복 체크
    )
    
    if success:
        print("데이터 업로드가 완료되었습니다.")
    else:
        print("데이터 업로드에 실패했습니다.")


if __name__ == "__main__":
    main()
