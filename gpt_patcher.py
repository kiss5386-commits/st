#!/usr/bin/env python3
"""
GPT 자동 코드 패치 도구 - 개선된 토큰 관리 버전
컨텍스트 길이 초과 문제를 해결하고 대용량 파일 처리 능력 강화
"""

import os
import json
import yaml
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import openai

# ===== 설정 =====
OPENAI_MODEL = "gpt-3.5-turbo"  # 기본 모델 (컨텍스트 길이 제한 대응)
MAX_CONTEXT_TOKENS = 6000       # 안전 마진 고려한 최대 토큰
MAX_COMPLETION_TOKENS = 2000    # 응답 토큰 제한
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

# 추적할 파일 패턴
TRACKED_PATTERNS = [
    "*.py", "*.js", "*.ts", "*.html", "*.css", "*.json", "*.yml", "*.yaml",
    "*.md", "*.txt", "*.sh", "*.bat", "*.sql", "*.env", "*.ini", "*.cfg"
]

# 무시할 디렉토리/파일
IGNORE_PATTERNS = [
    ".git", "__pycache__", "node_modules", ".venv", "venv", 
    "build", "dist", ".pytest_cache", ".idea", ".vscode"
]

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TokenManager:
    """토큰 사용량 관리 및 컨텍스트 최적화"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """텍스트의 대략적인 토큰 수 추정 (1토큰 ≈ 4글자)"""
        return len(text) // 3  # 보수적 추정
    
    @staticmethod
    def truncate_content(content: str, max_tokens: int) -> str:
        """컨텐츠를 토큰 제한에 맞게 잘라냄"""
        estimated_tokens = TokenManager.estimate_tokens(content)
        if estimated_tokens <= max_tokens:
            return content
        
        # 대략적 비율로 잘라내기
        ratio = max_tokens / estimated_tokens
        truncate_length = int(len(content) * ratio * 0.9)  # 안전 마진
        
        truncated = content[:truncate_length]
        return truncated + "\n\n[... 내용이 길어 일부 생략됨 ...]"
    
    @staticmethod
    def optimize_file_list(files: List[Dict], max_tokens: int) -> List[Dict]:
        """파일 목록을 토큰 제한에 맞게 최적화"""
        total_tokens = 0
        optimized_files = []
        
        # 작은 파일부터 우선 처리
        sorted_files = sorted(files, key=lambda f: len(f.get('content', '')))
        
        for file_info in sorted_files:
            content = file_info.get('content', '')
            file_tokens = TokenManager.estimate_tokens(content)
            
            if total_tokens + file_tokens > max_tokens:
                # 남은 토큰으로 파일 내용 축약
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # 최소 100토큰은 있어야 의미있음
                    file_info['content'] = TokenManager.truncate_content(content, remaining_tokens)
                    optimized_files.append(file_info)
                break
            
            optimized_files.append(file_info)
            total_tokens += file_tokens
        
        logger.info(f"📊 파일 최적화: {len(files)} → {len(optimized_files)}개, 예상 토큰: {total_tokens}")
        return optimized_files

class GitFileTracker:
    """Git 저장소 파일 추적 및 관리"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.tracking_file = self.repo_path / ".gpt_tracking.json"
    
    def should_ignore(self, path: Path) -> bool:
        """파일/디렉토리가 무시 대상인지 확인"""
        path_str = str(path)
        return any(pattern in path_str for pattern in IGNORE_PATTERNS)
    
    def get_all_tracked_files(self) -> List[Path]:
        """추적 대상 파일 목록 가져오기"""
        tracked_files = []
        
        for pattern in TRACKED_PATTERNS:
            for file_path in self.repo_path.rglob(pattern):
                if file_path.is_file() and not self.should_ignore(file_path):
                    tracked_files.append(file_path)
        
        return tracked_files
    
    def load_tracking_info(self) -> Dict:
        """기존 추적 정보 로드"""
        if not self.tracking_file.exists():
            return {}
        
        try:
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ 추적 파일 로드 실패: {e}")
            return {}
    
    def save_tracking_info(self, tracking_info: Dict):
        """추적 정보 저장"""
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(tracking_info, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 추적 정보 저장 완료: {len(tracking_info)}개 파일")
        except Exception as e:
            logger.error(f"❌ 추적 정보 저장 실패: {e}")
    
    def get_changed_files(self) -> List[Dict]:
        """변경된 파일 목록과 내용 반환"""
        current_files = self.get_all_tracked_files()
        tracking_info = self.load_tracking_info()
        changed_files = []
        
        for file_path in current_files:
            try:
                # 파일 내용 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                relative_path = str(file_path.relative_to(self.repo_path))
                file_hash = hash(content)
                
                # 변경 여부 확인
                if relative_path not in tracking_info or tracking_info[relative_path] != file_hash:
                    changed_files.append({
                        'path': relative_path,
                        'content': content,
                        'size': len(content),
                        'is_new': relative_path not in tracking_info
                    })
                    tracking_info[relative_path] = file_hash
            
            except Exception as e:
                logger.warning(f"⚠️ 파일 처리 실패 {file_path}: {e}")
        
        # 추적 정보 업데이트
        self.save_tracking_info(tracking_info)
        return changed_files

class GPTPatcher:
    """GPT 기반 자동 코드 패치"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.token_manager = TokenManager()
        logger.info("✅ OpenAI 클라이언트 초기화 완료")
    
    def create_system_prompt(self) -> str:
        """시스템 프롬프트 생성"""
        return """당신은 전문 코드 개발자입니다. 다음 규칙을 준수하세요:

1. **JSON 형식 응답**: 반드시 다음 형식으로 응답
```json
{
  "plan": "수행할 작업 계획",
  "files": [
    {
      "path": "파일경로",
      "action": "create|modify|delete",
      "content": "전체 파일 내용 (create/modify시)",
      "reason": "작업 이유"
    }
  ],
  "summary": "작업 요약"
}
```

2. **파일 처리 규칙**:
   - create: 새 파일 생성
   - modify: 기존 파일 수정 (전체 내용 덮어쓰기)
   - delete: 파일 삭제

3. **코드 품질**:
   - 에러 처리 필수 (try/except)
   - 로깅 추가
   - 주석으로 동작 설명
   - 기존 기능 보존

4. **안전장치**:
   - 중요 파일 수정시 백업 고려
   - 호환성 유지
   - 테스트 가능한 구조"""
    
    def create_user_prompt(self, instructions: str, files: List[Dict]) -> str:
        """사용자 프롬프트 생성"""
        # 파일 정보 요약
        files_summary = []
        for file_info in files:
            status = "새 파일" if file_info.get('is_new') else "수정됨"
            size_kb = file_info['size'] / 1024
            files_summary.append(f"- {file_info['path']} ({status}, {size_kb:.1f}KB)")
        
        prompt = f"""## 작업 지시문
{instructions}

## 현재 파일 상태
{chr(10).join(files_summary)}

## 파일 내용
"""
        
        # 파일 내용 추가
        for file_info in files[:10]:  # 최대 10개 파일만 포함
            prompt += f"""
### {file_info['path']}
```
{file_info['content']}
```
"""
        
        return prompt
    
    def call_gpt_api(self, instructions: str, files: List[Dict]) -> Optional[Dict]:
        """GPT API 호출"""
        # 토큰 최적화
        available_tokens = MAX_CONTEXT_TOKENS - 1000  # 시스템 프롬프트 등을 위한 여유
        optimized_files = self.token_manager.optimize_file_list(files, available_tokens)
        
        system_prompt = self.create_system_prompt()
        user_prompt = self.create_user_prompt(instructions, optimized_files)
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                logger.info(f"🤖 GPT API 호출 중... (시도 {attempt + 1}/{RETRY_ATTEMPTS})")
                
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=MAX_COMPLETION_TOKENS,
                    temperature=0.3
                )
                
                content = response.choices[0].message.content.strip()
                
                # JSON 파싱 시도
                try:
                    # 코드 블록이 있다면 제거
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    result = json.loads(content)
                    logger.info("✅ GPT API 호출 성공")
                    return result
                
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON 파싱 실패: {e}")
                    logger.error(f"응답 내용: {content[:500]}...")
                    return None
            
            except Exception as e:
                logger.warning(f"⚠️ 시도 {attempt + 1}: LLM 호출 실패 - {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
        
        logger.error("❌ ERROR: GPT API 호출 최종 실패")
        return None
    
    def execute_file_operations(self, operations: List[Dict]) -> bool:
        """파일 작업 실행"""
        success_count = 0
        
        for op in operations:
            try:
                path = Path(op['path'])
                action = op['action']
                
                if action == "create" or action == "modify":
                    # 디렉토리 생성
                    path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 파일 쓰기
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(op['content'])
                    
                    logger.info(f"✅ {action}: {path}")
                    success_count += 1
                
                elif action == "delete":
                    if path.exists():
                        path.unlink()
                        logger.info(f"🗑️ 삭제: {path}")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ 삭제 대상 없음: {path}")
                
            except Exception as e:
                logger.error(f"❌ 파일 작업 실패 {op['path']}: {e}")
        
        logger.info(f"📊 파일 작업 완료: {success_count}/{len(operations)} 성공")
        return success_count > 0

def main():
    """메인 실행 함수"""
    try:
        # 환경 변수 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다")
            return False
        
        instructions = os.getenv("USER_INSTRUCTIONS", "")
        if not instructions:
            logger.error("❌ USER_INSTRUCTIONS 환경변수가 설정되지 않았습니다")
            return False
        
        logger.info("🚀 GPT 자동 코드 수정 시작")
        logger.info("=" * 50)
        logger.info(f"📝 사용자 지시문: {instructions[:100]}...")
        
        # 파일 추적 시작
        tracker = GitFileTracker()
        changed_files = tracker.get_changed_files()
        
        if not changed_files:
            logger.info("📂 변경된 파일이 없습니다. 기본 테스트 파일을 생성합니다.")
            # 기본 테스트 파일 생성
            test_content = f"""# GPT 자동 수정 테스트
# 생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}
# 지시문: {instructions[:100]}...

print("GPT 자동 수정 테스트 성공!")
"""
            changed_files = [{
                'path': f'gpt_test_{int(time.time())}.py',
                'content': test_content,
                'size': len(test_content),
                'is_new': True
            }]
        
        logger.info(f"📂 처리할 파일 {len(changed_files)}개 확인")
        
        # GPT 패치 실행
        patcher = GPTPatcher(api_key)
        result = patcher.call_gpt_api(instructions, changed_files)
        
        if not result:
            logger.error("❌ GPT API 호출 실패")
            return False
        
        # 파일 작업 실행
        if 'files' in result and result['files']:
            success = patcher.execute_file_operations(result['files'])
            
            if success:
                logger.info("✅ 모든 작업이 성공적으로 완료되었습니다")
                logger.info(f"📋 작업 요약: {result.get('summary', 'N/A')}")
                return True
            else:
                logger.error("❌ 파일 작업 중 오류 발생")
                return False
        else:
            logger.warning("⚠️ 수행할 파일 작업이 없습니다")
            return True
    
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
