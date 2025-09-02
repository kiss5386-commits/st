#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # 환경변수로 모델 선택
MAX_CONTEXT_TOKENS = 200000     # GPT-5 기준 대용량 컨텍스트
MAX_COMPLETION_TOKENS = 16000   # GPT-5 응답 토큰 확대
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

# 모델별 토큰 제한 설정
MODEL_LIMITS = {
    "gpt-5": {"context": 200000, "completion": 16000},
    "gpt-5-mini": {"context": 200000, "completion": 8000},
    "gpt-5-nano": {"context": 128000, "completion": 4000},
    "gpt-4o": {"context": 120000, "completion": 8000}, 
    "gpt-4o-mini": {"context": 120000, "completion": 8000},
    "o1-preview": {"context": 120000, "completion": 32000},
    "o1-mini": {"context": 120000, "completion": 16000}
}

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
    def truncate_content(content: str, max_tokens: int) -> str:
        """컨텐츠를 토큰 제한에 맞게 잘라냄"""
        estimated_tokens = TokenManager.estimate_tokens(content)
        if estimated_tokens <= max_tokens:
            return content

        # 대략적 비율로 잘라내기(+안전 마진)
        ratio = max_tokens / max(estimated_tokens, 1)
        truncate_length = int(len(content) * ratio * 0.9)

        truncated = content[:truncate_length]
        return truncated + "\n\n[... 내용이 길어 일부 생략됨 ...]"
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """텍스트의 대략적인 토큰 수 추정 (1토큰 ≈ 4글자)"""
        return len(text) // 3  # 보수적 추정
    
    @staticmethod
    def extract_code_structure(content: str, max_tokens: int) -> str:
        """대용량 코드에서 핵심 구조만 추출"""
        lines = content.split('\n')
        extracted_lines = []
        current_tokens = 0
        
        # 우선순위별 추출
        priorities = [
            # 1순위: 클래스/함수 정의
            (r'^class\s+\w+', '클래스 정의'),
            (r'^def\s+\w+', '함수 정의'),
            (r'^\s+def\s+\w+', '메서드 정의'),
            # 2순위: 중요 설정/상수
            (r'^[A-Z_]+=', '상수 정의'),
            (r'^import\s+|^from\s+', '임포트'),
            # 3순위: 주석과 독스트링
            (r'^\s*"""', '독스트링'),
            (r'^\s*#.*중요|^\s*#.*TODO|^\s*#.*FIXME', '중요 주석'),
        ]
        
        import re
        
        # 우선순위별 추출
        for pattern, desc in priorities:
            if current_tokens >= max_tokens * 0.9:
                break
                
            for i, line in enumerate(lines):
                if re.match(pattern, line):
                    # 함수/클래스의 경우 시그니처와 독스트링 포함
                    if 'def ' in line or 'class ' in line:
                        block = TokenManager.extract_function_block(lines, i)
                        block_tokens = TokenManager.estimate_tokens('\n'.join(block))
                        
                        if current_tokens + block_tokens <= max_tokens:
                            extracted_lines.extend(block)
                            current_tokens += block_tokens
                    else:
                        line_tokens = TokenManager.estimate_tokens(line)
                        if current_tokens + line_tokens <= max_tokens:
                            extracted_lines.append(line)
                            current_tokens += line_tokens
        
        if not extracted_lines:
            # 구조 추출 실패시 앞부분만
            truncated = TokenManager.truncate_content(content, max_tokens)
            return truncated
        
        result = '\n'.join(extracted_lines)
        result += f"\n\n# ... 총 {len(lines)}줄 중 핵심 구조 {len(extracted_lines)}줄 추출됨 ..."
        return result
    
    @staticmethod
    def extract_function_block(lines: List[str], start_idx: int) -> List[str]:
        """함수/클래스 블록 추출 (시그니처 + 독스트링만)"""
        block = [lines[start_idx]]  # 함수/클래스 정의 라인
        
        i = start_idx + 1
        indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        # 독스트링이나 주요 주석만 포함
        in_docstring = False
        docstring_quotes = None
        
        while i < len(lines) and i < start_idx + 20:  # 최대 20줄까지만
            line = lines[i]
            
            if not line.strip():  # 빈 줄
                block.append(line)
                i += 1
                continue
                
            line_indent = len(line) - len(line.lstrip())
            
            # 들여쓰기가 원래보다 작거나 같으면 함수 끝
            if line_indent <= indent_level and line.strip():
                break
            
            # 독스트링 처리
            if '"""' in line or "'''" in line:
                if not in_docstring:
                    docstring_quotes = '"""' if '"""' in line else "'''"
                    in_docstring = True
                    block.append(line)
                elif docstring_quotes in line:
                    block.append(line)
                    in_docstring = False
                    break  # 독스트링 끝나면 함수 시그니처 완료
                else:
                    block.append(line)
            elif in_docstring:
                block.append(line)
            elif line.strip().startswith('#'):  # 주석
                block.append(line)
            elif 'return ' in line or 'yield ' in line:  # 반환 타입 힌트
                block.append(line)
                break
            
            i += 1
        
        # 함수 본문 생략 표시
        if i < len(lines) and i > start_idx + 1:
            block.append(" " * (indent_level + 4) + "# ... 함수 본문 생략 ...")
            
        return block
    
    @staticmethod
    def optimize_file_list(files: List[Dict], max_tokens: int) -> List[Dict]:
        """파일 목록을 토큰 제한에 맞게 최적화 - 대용량 단일파일 지원"""
        total_tokens = 0
        optimized_files = []
        
        # 우선순위: .py 파일 먼저, 그 다음 크기순
        def file_priority(f):
            path = f.get('path', '')
            if 'stargate' in path and path.endswith('.py'):
                return 0  # 최고 우선순위
            elif path.endswith('.py'):
                return 1
            else:
                return 2
        
        sorted_files = sorted(files, key=lambda f: (file_priority(f), len(f.get('content', ''))))
        
        for file_info in sorted_files:
            content = file_info.get('content', '')
            file_tokens = TokenManager.estimate_tokens(content)
            path = file_info.get('path', '')
            
            # 대용량 단일 파일 특별 처리
            if file_tokens > max_tokens * 0.8:  # 전체 토큰의 80% 이상
                logger.info(f"🔥 대용량 파일 감지: {path} ({file_tokens:,} 토큰)")
                
                # 핵심 부분만 추출 (함수 시그니처, 클래스 정의, 주요 로직)
                optimized_content = TokenManager.extract_code_structure(content, max_tokens)
                file_info['content'] = optimized_content
                file_info['is_large_file'] = True
                optimized_files.append(file_info)
                total_tokens += TokenManager.estimate_tokens(optimized_content)
                
                logger.info(f"📝 대용량 파일 최적화: {file_tokens:,} → {TokenManager.estimate_tokens(optimized_content):,} 토큰")
                break  # 대용량 파일 하나만 처리
            
            # 일반 파일 처리
            elif total_tokens + file_tokens <= max_tokens:
                optimized_files.append(file_info)
                total_tokens += file_tokens
            else:
                # 남은 공간으로 축약
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 500:  # 최소 500토큰
                    file_info['content'] = TokenManager.truncate_content(content, remaining_tokens)
                    optimized_files.append(file_info)
                    total_tokens = max_tokens
                break
        
        logger.info(f"📊 파일 최적화: {len(files)} → {len(optimized_files)}개, 예상 토큰: {total_tokens:,}")
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
        self.model = OPENAI_MODEL
        
        # 모델별 제한 설정
        if self.model in MODEL_LIMITS:
            self.max_context = MODEL_LIMITS[self.model]["context"] - 2000  # 안전 마진
            self.max_completion = MODEL_LIMITS[self.model]["completion"]
        else:
            # 기본값 (안전한 설정)
            self.max_context = 6000
            self.max_completion = 2000
            
        logger.info(f"✅ OpenAI 클라이언트 초기화 완료 (모델: {self.model})")
        logger.info(f"📊 토큰 제한: Context={self.max_context}, Completion={self.max_completion}")
    
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
        # 동적 토큰 최적화
        available_tokens = self.max_context - 1000  # 시스템 프롬프트 등을 위한 여유
        optimized_files = self.token_manager.optimize_file_list(files, available_tokens)
        
        system_prompt = self.create_system_prompt()
        user_prompt = self.create_user_prompt(instructions, optimized_files)
        
        # o1 모델 계열은 다른 파라미터 사용
        is_o1_model = self.model.startswith("o1")
        is_gpt5_model = self.model.startswith("gpt-5")
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                logger.info(f"🤖 GPT API 호출 중... (시도 {attempt + 1}/{RETRY_ATTEMPTS})")
                
                if is_o1_model:
                    # o1 모델은 system 메시지를 지원하지 않음
                    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": combined_prompt}
                        ],
                        max_completion_tokens=self.max_completion
                    )
                elif is_gpt5_model:
                    # GPT-5는 향상된 파라미터 지원
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=self.max_completion,
                        temperature=0.2,  # GPT-5는 더 낮은 온도로 정확성 향상
                        top_p=0.9
                    )
                else:
                    # 일반 모델 (GPT-4o 등)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=self.max_completion,
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
