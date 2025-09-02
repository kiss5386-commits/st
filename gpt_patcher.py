#!/usr/bin/env python3
"""
GPT 기반 자동 코드 수정 스크립트
- 시드 보호 최우선: 안전 경로만 수정, 위험 경로는 절대 차단
- unified diff 사용 금지: JSON 직접 적용 방식으로 패치 오류 원천 차단
- 기존 파일 존재 여부 검증: new file 오류 방지
- 환경변수 전용 입력: 셸 파싱 오류 방지
"""

import os
import sys
import json
import subprocess
import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

# 보안 설정 - 시드 보호를 위한 경로 제한
SAFE_PATHS = [
    "app", 
    "server", 
    "configs", 
    "stargate_all_in_one.py",
    "strategies",
    "utils",
    "tests",
    "docs"
]

DENY_PATHS = [
    ".github",
    "secrets", 
    "keys", 
    "certs",
    ".git",
    "node_modules",
    "__pycache__",
    ".env"
]

MAX_RETRIES = 3
OPENAI_TEMPERATURE = 0.1

class GPTPatcher:
    """
    GPT API를 통한 안전한 코드 자동 수정 클래스
    - 패치 실패 원인들을 모두 해결한 견고한 구현
    - 시드 보호를 위한 다단계 보안 검증
    """
    
    def __init__(self):
        self.client = None
        self.run_id = os.environ.get('GITHUB_RUN_ID', 'local')
        self.setup_openai()
        
    def setup_openai(self) -> bool:
        """OpenAI 클라이언트 초기화"""
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print("❌ ERROR: OPENAI_API_KEY 환경변수가 설정되지 않았습니다")
                return False
                
            self.client = OpenAI(api_key=api_key)
            print("✅ OpenAI 클라이언트 초기화 완료")
            return True
            
        except Exception as e:
            print(f"❌ ERROR: OpenAI 초기화 실패 - {e}")
            return False

    def get_user_instructions(self) -> Optional[str]:
        """환경변수에서 사용자 지시문 추출 - 셸 파싱 오류 방지"""
        instructions = os.environ.get('USER_INSTRUCTIONS', '').strip()
        if not instructions:
            print("❌ ERROR: USER_INSTRUCTIONS 환경변수가 비어있습니다")
            return None
            
        print(f"📝 사용자 지시문: {instructions[:100]}{'...' if len(instructions) > 100 else ''}")
        return instructions

    def get_existing_files(self) -> List[str]:
        """현재 git에서 추적 중인 파일 목록 조회 - new file 오류 방지"""
        try:
            result = subprocess.run(
                ['git', 'ls-files'],
                capture_output=True,
                text=True,
                check=True
            )
            files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            print(f"📂 기존 추적 파일 {len(files)}개 확인")
            return files
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ WARNING: git ls-files 실행 실패 - {e}")
            return []
    
    def is_path_safe(self, path: str) -> Tuple[bool, str]:
        """경로 보안 검증 - 시드 보호를 위한 핵심 기능"""
        path = path.strip().replace('\\', '/')
        
        # 절대 경로나 상위 디렉토리 접근 차단
        if path.startswith('/') or '..' in path:
            return False, "절대경로 또는 상위디렉토리 접근 시도"
            
        # 금지 경로 검사
        for deny in DENY_PATHS:
            if path.startswith(deny + '/') or path == deny:
                return False, f"금지 경로 [{deny}] 접근 시도"
        
        # 안전 경로 검사
        for safe in SAFE_PATHS:
            if path.startswith(safe + '/') or path == safe:
                return True, "안전 경로 확인"
        
        return False, "허용된 안전 경로 범위 밖"

    def create_llm_prompt(self, instructions: str, existing_files: List[str]) -> Dict[str, str]:
        """LLM용 프롬프트 생성 - unified diff 사용 절대 금지"""
        system_prompt = f"""너는 암호화폐 자동매매 시스템의 전문 코드 수정 어시스턴트다.

⚠️ 절대 규칙:
1. 출력은 오직 JSON 형식만 허용 (설명, 코드펜스, unified diff 절대 금지)
2. 기존 파일은 절대 new file로 생성하지 말고 반드시 update로 처리
3. 안전 경로만 수정 가능: {SAFE_PATHS}
4. 금지 경로 절대 수정 금지: {DENY_PATHS}
5. 시드 보호 로직(손절, 트레일링, 쿨다운)의 기본값이나 핵심 흐름은 변경 금지
6. 불분명한 요청은 TODO 주석만 추가

출력 형식 (엄격 준수):
{{
  "files": [
    {{
      "path": "경로/파일명",
      "op": "create|update|delete",
      "content": "파일 전체 내용 (delete시 빈 문자열)"
    }}
  ]
}}"""

        user_prompt = f"""🎯 사용자 지시문:
{instructions}

📋 현재 상황:
- 안전 경로: {SAFE_PATHS}
- 금지 경로: {DENY_PATHS}
- 기존 추적 파일: {len(existing_files)}개

기존 파일 목록:
{chr(10).join(existing_files[:50])}
{'... (더 많은 파일 생략)' if len(existing_files) > 50 else ''}

⚠️ 중요: 위 기존 파일들은 절대 "create"하지 말고 "update"만 사용하라.

JSON 스키마만 반환 (설명/코드펜스 절대 금지):
{{
  "files": [
    {{ "path": "...", "op": "create|update|delete", "content": "..." }}
  ]
}}"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def call_llm(self, prompts: Dict[str, str]) -> Optional[Dict]:
        """LLM 호출 및 응답 처리 - 재시도 로직 포함"""
        for attempt in range(MAX_RETRIES):
            try:
                print(f"🤖 GPT API 호출 중... (시도 {attempt + 1}/{MAX_RETRIES})")
                
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": prompts["system"]},
                        {"role": "user", "content": prompts["user"]}
                    ],
                    temperature=OPENAI_TEMPERATURE,
                    max_tokens=8000
                )
                
                content = response.choices[0].message.content.strip()
                
                # JSON 추출 - 코드펜스나 추가 텍스트 제거
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if not json_match:
                    print(f"⚠️ 시도 {attempt + 1}: JSON 형식을 찾을 수 없음")
                    continue
                
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # 기본 구조 검증
                if not isinstance(result, dict) or 'files' not in result:
                    print(f"⚠️ 시도 {attempt + 1}: 잘못된 JSON 구조")
                    continue
                
                if not isinstance(result['files'], list):
                    print(f"⚠️ 시도 {attempt + 1}: files가 배열이 아님")
                    continue
                
                print(f"✅ GPT 응답 파싱 성공 - {len(result['files'])}개 파일 작업")
                return result
                
            except json.JSONDecodeError as e:
                print(f"⚠️ 시도 {attempt + 1}: JSON 파싱 실패 - {e}")
                if attempt == MAX_RETRIES - 1:
                    print(f"📄 마지막 응답 내용: {content[:500]}...")
                    
            except Exception as e:
                print(f"⚠️ 시도 {attempt + 1}: LLM 호출 실패 - {e}")
        
        print("❌ ERROR: GPT API 호출 최종 실패")
        return None

    def validate_and_fix_operations(self, files_data: List[Dict], existing_files: List[str]) -> List[Dict]:
        """작업 데이터 검증 및 자동 보정"""
        valid_operations = []
        existing_set = set(existing_files)
        
        for file_data in files_data:
            if not isinstance(file_data, dict):
                print(f"⚠️ SKIP: 잘못된 파일 데이터 형식 - {file_data}")
                continue
            
            path = file_data.get('path', '').strip()
            op = file_data.get('op', '').strip().lower()
            content = file_data.get('content', '')
            
            if not path or op not in ['create', 'update', 'delete']:
                print(f"⚠️ SKIP: 잘못된 경로 또는 작업 - path:{path}, op:{op}")
                continue
            
            # 보안 검증
            is_safe, reason = self.is_path_safe(path)
            if not is_safe:
                print(f"🚫 DENY: {path} - {reason}")
                continue
            
            # create/update 자동 보정
            if op == 'create' and path in existing_set:
                print(f"🔧 AUTO-FIX: {path} - create → update (기존 파일 존재)")
                op = 'update'
            
            # delete 작업 제한 (선택적 - 안전성 강화)
            if op == 'delete':
                print(f"⚠️ DELETE 요청: {path} - 신중히 처리")
            
            valid_operations.append({
                'path': path,
                'op': op,
                'content': content
            })
            print(f"✅ VALID: {path} [{op}]")
        
        return valid_operations

    def apply_file_operations(self, operations: List[Dict]) -> bool:
        """파일 작업 적용 - unified diff 대신 직접 파일 조작"""
        if not operations:
            print("📭 적용할 파일 작업이 없습니다")
            return True
        
        success_count = 0
        
        for op_data in operations:
            path = op_data['path']
            op = op_data['op']
            content = op_data['content']
            
            try:
                if op == 'delete':
                    if os.path.exists(path):
                        # Git에서 파일 제거
                        subprocess.run(['git', 'rm', path], check=True, capture_output=True)
                        print(f"🗑️ DELETED: {path}")
                        success_count += 1
                    else:
                        print(f"⚠️ SKIP DELETE: {path} - 파일이 존재하지 않음")
                        
                else:  # create 또는 update
                    # 디렉토리 생성
                    dir_path = os.path.dirname(path)
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)
                    
                    # 파일 쓰기
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Git에 추가
                    subprocess.run(['git', 'add', path], check=True, capture_output=True)
                    
                    action = "CREATED" if op == 'create' else "UPDATED"
                    print(f"📝 {action}: {path} ({len(content)} chars)")
                    success_count += 1
                    
            except Exception as e:
                print(f"❌ ERROR applying {op} to {path}: {e}")
                return False
        
        print(f"✅ 파일 작업 완료: {success_count}/{len(operations)}")
        return True

    def check_changes_and_commit(self) -> bool:
        """변경사항 확인 및 커밋"""
        try:
            # 변경사항 확인
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            
            changes = result.stdout.strip()
            if not changes:
                print("📭 변경사항이 없습니다 - PR 생성하지 않음")
                return True
            
            print(f"📋 감지된 변경사항:\n{changes}")
            
            # 브랜치 생성 및 체크아웃
            branch_name = f"gpt/change-{self.run_id}"
            subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
            print(f"🌿 브랜치 생성: {branch_name}")
            
            # 커밋
            commit_msg = f"GPT: Apply automated changes (Run #{self.run_id})"
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            print(f"💾 커밋 완료: {commit_msg}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Git 작업 실패 - {e}")
            return False

    def run(self) -> bool:
        """메인 실행 함수 - 전체 파이프라인 조율"""
        print("🚀 GPT 자동 코드 수정 시작")
        print("=" * 50)
        
        # 1. 입력 데이터 수집
        instructions = self.get_user_instructions()
        if not instructions:
            return False
        
        existing_files = self.get_existing_files()
        
        # 2. LLM 프롬프트 생성 및 호출
        prompts = self.create_llm_prompt(instructions, existing_files)
        llm_result = self.call_llm(prompts)
        if not llm_result:
            return False
        
        # 3. 작업 검증 및 보정
        operations = self.validate_and_fix_operations(
            llm_result.get('files', []), 
            existing_files
        )
        
        # 4. 파일 작업 적용
        if not self.apply_file_operations(operations):
            return False
        
        # 5. 변경사항 확인 및 커밋
        if not self.check_changes_and_commit():
            return False
        
        print("=" * 50)
        print("🎉 GPT 자동 코드 수정 완료!")
        return True


def main():
    """메인 엔트리 포인트"""
    try:
        patcher = GPTPatcher()
        success = patcher.run()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단됨")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
