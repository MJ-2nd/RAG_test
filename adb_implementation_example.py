#!/usr/bin/env python3
"""
ADB (Android Debug Bridge) Implementation Example
사용자가 control_android_device 도구를 구현할 때 참고할 수 있는 예시 코드입니다.
이 코드를 llm/app.py의 execute_tool 메서드에 통합하세요.
"""

import subprocess
import json
import os
import time
from typing import Dict, Any, Optional, List

class AndroidDeviceController:
    """Android 디바이스 제어를 위한 클래스"""
    
    def __init__(self, default_timeout: int = 30):
        """
        AndroidDeviceController 초기화
        
        Args:
            default_timeout (int): 기본 명령어 타임아웃 (초)
        """
        self.default_timeout = default_timeout
        self._connected_devices = None
        self._last_device_check = 0
        self._device_cache_duration = 5  # 5초간 디바이스 목록 캐시
        
    def check_adb_installed(self) -> bool:
        """ADB가 설치되어 있는지 확인"""
        try:
            result = subprocess.run(['adb', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_connected_devices(self, force_refresh: bool = False) -> List[str]:
        """
        연결된 Android 디바이스 목록 반환
        
        Args:
            force_refresh (bool): 캐시를 무시하고 새로 조회할지 여부
            
        Returns:
            List[str]: 연결된 디바이스 ID 목록
        """
        current_time = time.time()
        
        # 캐시된 결과가 있고, 캐시가 유효한 경우
        if (not force_refresh and 
            self._connected_devices is not None and 
            current_time - self._last_device_check < self._device_cache_duration):
            return self._connected_devices
        
        try:
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # 헤더 제외
                devices = []
                for line in lines:
                    if line.strip() and '\tdevice' in line:
                        device_id = line.split('\t')[0]
                        devices.append(device_id)
                
                # 캐시 업데이트
                self._connected_devices = devices
                self._last_device_check = current_time
                return devices
            return []
        except subprocess.TimeoutExpired:
            return []
    
    def execute_adb_command(self, command: List[str], device_id: Optional[str] = None, 
                           timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        ADB 명령어 실행
        
        Args:
            command (List[str]): 실행할 ADB 명령어 리스트
            device_id (Optional[str]): 대상 디바이스 ID (None이면 기본 디바이스)
            timeout (Optional[int]): 타임아웃 (초), None이면 기본값 사용
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        if timeout is None:
            timeout = self.default_timeout
            
        try:
            # 디바이스 ID가 지정된 경우 -s 옵션 추가
            if device_id:
                adb_cmd = ['adb', '-s', device_id] + command
            else:
                adb_cmd = ['adb'] + command
            
            result = subprocess.run(adb_cmd, 
                                  capture_output=True, text=True, timeout=timeout)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": ' '.join(adb_cmd)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "command": ' '.join(adb_cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": ' '.join(adb_cmd)
            }
    
    def tap(self, x: int, y: int, device_id: Optional[str] = None, 
            timeout: Optional[int] = None) -> Dict[str, Any]:
        """화면 터치"""
        command = ['shell', 'input', 'tap', str(x), str(y)]
        return self.execute_adb_command(command, device_id, timeout)
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, 
              device_id: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """스와이프"""
        command = ['shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2)]
        return self.execute_adb_command(command, device_id, timeout)
    
    def input_text(self, text: str, device_id: Optional[str] = None, 
                   timeout: Optional[int] = None) -> Dict[str, Any]:
        """텍스트 입력"""
        # 특수 문자 이스케이프 처리
        escaped_text = text.replace(' ', '%s').replace('&', '\\&')
        command = ['shell', 'input', 'text', escaped_text]
        return self.execute_adb_command(command, device_id, timeout)
    
    def keyevent(self, keycode: str, device_id: Optional[str] = None, 
                 timeout: Optional[int] = None) -> Dict[str, Any]:
        """키 이벤트"""
        command = ['shell', 'input', 'keyevent', keycode]
        return self.execute_adb_command(command, device_id, timeout)
    
    def start_app(self, package_name: str, activity_name: str = "", 
                  device_id: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """앱 실행"""
        if activity_name:
            app_component = f"{package_name}/{activity_name}"
        else:
            # 기본 런처 액티비티 시작
            app_component = package_name
        
        command = ['shell', 'am', 'start', '-n', app_component]
        return self.execute_adb_command(command, device_id, timeout)
    
    def list_packages(self, device_id: Optional[str] = None, 
                      timeout: Optional[int] = None) -> Dict[str, Any]:
        """설치된 패키지 목록"""
        command = ['shell', 'pm', 'list', 'packages']
        result = self.execute_adb_command(command, device_id, timeout)
        
        if result.get('success'):
            # 패키지 목록 파싱
            packages = []
            for line in result['stdout'].split('\n'):
                if line.startswith('package:'):
                    packages.append(line.replace('package:', '').strip())
            result['packages'] = packages
        
        return result
    
    def get_device_info(self, shell_command: str = "dumpsys battery", 
                        device_id: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """디바이스 정보 조회"""
        command = ['shell'] + shell_command.split()
        return self.execute_adb_command(command, device_id, timeout)
    
    def execute_shell_command(self, shell_command: str, device_id: Optional[str] = None, 
                             timeout: Optional[int] = None) -> Dict[str, Any]:
        """사용자 정의 쉘 명령어 실행"""
        command = ['shell'] + shell_command.split()
        return self.execute_adb_command(command, device_id, timeout)
    
    def take_screenshot(self, file_path: str = "/sdcard/screenshot.png", 
                        device_id: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """스크린샷 촬영"""
        command = ['shell', 'screencap', file_path]
        return self.execute_adb_command(command, device_id, timeout)
    
    def pull_file(self, remote_path: str, local_path: str, 
                  device_id: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """파일 다운로드"""
        command = ['pull', remote_path, local_path]
        return self.execute_adb_command(command, device_id, timeout)
    
    def push_file(self, local_path: str, remote_path: str, 
                  device_id: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """파일 업로드"""
        command = ['push', local_path, remote_path]
        return self.execute_adb_command(command, device_id, timeout)
    
    def get_screen_resolution(self, device_id: Optional[str] = None, 
                             timeout: Optional[int] = None) -> Dict[str, Any]:
        """화면 해상도 조회"""
        command = ['shell', 'wm', 'size']
        return self.execute_adb_command(command, device_id, timeout)
    
    def clear_app_data(self, package_name: str, device_id: Optional[str] = None, 
                       timeout: Optional[int] = None) -> Dict[str, Any]:
        """앱 데이터 초기화"""
        command = ['shell', 'pm', 'clear', package_name]
        return self.execute_adb_command(command, device_id, timeout)
    
    def install_apk(self, apk_path: str, device_id: Optional[str] = None, 
                    timeout: Optional[int] = None) -> Dict[str, Any]:
        """APK 설치"""
        command = ['install', apk_path]
        return self.execute_adb_command(command, device_id, timeout)
    
    def uninstall_app(self, package_name: str, device_id: Optional[str] = None, 
                      timeout: Optional[int] = None) -> Dict[str, Any]:
        """앱 제거"""
        command = ['uninstall', package_name]
        return self.execute_adb_command(command, device_id, timeout)
    
    def control_android_device(self, tool_call_arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool-calling을 위한 통합 메서드
        이 메서드를 llm/app.py의 execute_tool 메서드에서 호출하세요.
        
        Args:
            tool_call_arguments (Dict[str, Any]): 도구 호출 인자
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        
        # 1. ADB 설치 확인
        if not self.check_adb_installed():
            return {"error": "ADB is not installed or not accessible"}
        
        # 2. 파라미터 추출
        command_type = tool_call_arguments.get('command_type', 'custom_shell')
        device_id = tool_call_arguments.get('device_id')
        timeout = tool_call_arguments.get('timeout', self.default_timeout)
        
        # 3. 연결된 디바이스 확인
        devices = self.get_connected_devices()
        if not devices:
            return {"error": "No Android devices connected"}
        
        # 디바이스 ID가 지정되지 않은 경우 첫 번째 디바이스 사용
        if not device_id and devices:
            device_id = devices[0]
        
        # 4. 명령어 타입별 처리
        try:
            if command_type == "tap":
                coordinates = tool_call_arguments.get('coordinates', {})
                x = coordinates.get('x')
                y = coordinates.get('y')
                if x is None or y is None:
                    return {"error": "Coordinates (x, y) required for tap command"}
                
                result = self.tap(x, y, device_id, timeout)
                
            elif command_type == "swipe":
                coordinates = tool_call_arguments.get('coordinates', {})
                x1, y1 = coordinates.get('x'), coordinates.get('y')
                x2, y2 = coordinates.get('x2'), coordinates.get('y2')
                if None in [x1, y1, x2, y2]:
                    return {"error": "Coordinates (x, y, x2, y2) required for swipe command"}
                
                result = self.swipe(x1, y1, x2, y2, device_id, timeout)
                
            elif command_type == "input_text":
                text = tool_call_arguments.get('text')
                if not text:
                    return {"error": "Text parameter required for input_text command"}
                
                result = self.input_text(text, device_id, timeout)
                
            elif command_type == "keyevent":
                keycode = tool_call_arguments.get('keycode')
                if not keycode:
                    return {"error": "Keycode parameter required for keyevent command"}
                
                result = self.keyevent(keycode, device_id, timeout)
                
            elif command_type == "start_app":
                package_name = tool_call_arguments.get('package_name')
                activity_name = tool_call_arguments.get('activity_name', '')
                if not package_name:
                    return {"error": "Package name required for start_app command"}
                
                result = self.start_app(package_name, activity_name, device_id, timeout)
                
            elif command_type == "list_packages":
                result = self.list_packages(device_id, timeout)
                
            elif command_type == "get_info":
                shell_command = tool_call_arguments.get('shell_command', 'dumpsys battery')
                result = self.get_device_info(shell_command, device_id, timeout)
                
            elif command_type == "custom_shell":
                shell_command = tool_call_arguments.get('shell_command')
                if not shell_command:
                    return {"error": "Shell command required for custom_shell command"}
                
                result = self.execute_shell_command(shell_command, device_id, timeout)
                
            else:
                return {"error": f"Unsupported command type: {command_type}"}
            
            # 5. 결과 반환
            if result.get('success'):
                return {
                    "result": result['stdout'].strip(),
                    "command_type": command_type,
                    "device_id": device_id,
                    "success": True
                }
            else:
                return {
                    "error": result.get('error', result.get('stderr', 'Unknown error')),
                    "command_type": command_type,
                    "device_id": device_id,
                    "success": False
                }
                
        except Exception as e:
            return {"error": f"Android device control error: {str(e)}"}

# 사용 예시
if __name__ == "__main__":
    # 컨트롤러 인스턴스 생성
    controller = AndroidDeviceController(default_timeout=30)
    
    # 예시 1: 화면 터치
    tap_example = {
        "command_type": "tap",
        "coordinates": {"x": 500, "y": 1000}
    }
    
    # 예시 2: 텍스트 입력
    text_example = {
        "command_type": "input_text",
        "text": "Hello Android"
    }
    
    # 예시 3: 홈 버튼 누르기
    home_example = {
        "command_type": "keyevent",
        "keycode": "KEYCODE_HOME"
    }
    
    # 예시 4: 앱 실행
    app_example = {
        "command_type": "start_app",
        "package_name": "com.android.settings"
    }
    
    # 예시 5: 사용자 정의 명령어
    custom_example = {
        "command_type": "custom_shell",
        "shell_command": "pm list packages | grep chrome"
    }
    
    print("AndroidDeviceController - Ready for integration!")
    print("Usage example:")
    print("controller = AndroidDeviceController()")
    print("result = controller.control_android_device(tool_call_arguments)") 