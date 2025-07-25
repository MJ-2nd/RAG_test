#!/usr/bin/env python3
"""
AndroidDeviceController 테스트 스크립트
클래스의 다양한 기능을 테스트하고 사용법을 보여줍니다.
"""

import sys
import time
from adb_implementation_example import AndroidDeviceController

def test_basic_functionality():
    """기본 기능 테스트"""
    print("🔧 AndroidDeviceController 기본 기능 테스트")
    print("=" * 50)
    
    # 컨트롤러 생성
    controller = AndroidDeviceController(default_timeout=30)
    
    # 1. ADB 설치 확인
    print("1. ADB 설치 확인...")
    if controller.check_adb_installed():
        print("   ✅ ADB가 설치되어 있습니다.")
    else:
        print("   ❌ ADB가 설치되어 있지 않습니다.")
        print("   Ubuntu/Debian: sudo apt install android-tools-adb")
        print("   macOS: brew install android-platform-tools")
        return False
    
    # 2. 연결된 디바이스 확인
    print("\n2. 연결된 디바이스 확인...")
    devices = controller.get_connected_devices()
    if devices:
        print(f"   ✅ 연결된 디바이스: {devices}")
    else:
        print("   ❌ 연결된 Android 디바이스가 없습니다.")
        print("   - USB 디버깅을 활성화하세요")
        print("   - 디바이스를 USB로 연결하세요")
        return False
    
    return True

def test_individual_methods():
    """개별 메서드 테스트"""
    print("\n🔧 개별 메서드 테스트")
    print("=" * 50)
    
    controller = AndroidDeviceController()
    
    # 기본 디바이스 ID (첫 번째 디바이스 사용)
    devices = controller.get_connected_devices()
    device_id = devices[0] if devices else None
    
    if not device_id:
        print("❌ 테스트할 디바이스가 없습니다.")
        return
    
    print(f"테스트 대상 디바이스: {device_id}")
    
    # 1. 화면 해상도 조회
    print("\n1. 화면 해상도 조회...")
    result = controller.get_screen_resolution(device_id)
    if result.get('success'):
        print(f"   ✅ 해상도: {result['stdout'].strip()}")
    else:
        print(f"   ❌ 실패: {result.get('error', 'Unknown error')}")
    
    # 2. 배터리 정보 조회
    print("\n2. 배터리 정보 조회...")
    result = controller.get_device_info("dumpsys battery", device_id)
    if result.get('success'):
        print("   ✅ 배터리 정보 조회 성공")
        # 배터리 정보에서 주요 정보만 추출
        output = result['stdout']
        for line in output.split('\n'):
            if any(keyword in line.lower() for keyword in ['level', 'status', 'powered']):
                print(f"      {line.strip()}")
    else:
        print(f"   ❌ 실패: {result.get('error', 'Unknown error')}")
    
    # 3. 설치된 패키지 목록 (일부만)
    print("\n3. 설치된 패키지 목록 (최대 10개)...")
    result = controller.list_packages(device_id)
    if result.get('success'):
        packages = result.get('packages', [])
        print(f"   ✅ 총 {len(packages)}개 패키지 발견")
        for i, package in enumerate(packages[:10]):
            print(f"      {i+1}. {package}")
        if len(packages) > 10:
            print(f"      ... 및 {len(packages) - 10}개 더")
    else:
        print(f"   ❌ 실패: {result.get('error', 'Unknown error')}")

def test_tool_calling_integration():
    """Tool-calling 통합 테스트"""
    print("\n🔧 Tool-calling 통합 테스트")
    print("=" * 50)
    
    controller = AndroidDeviceController()
    
    # 테스트할 명령어들
    test_commands = [
        {
            "name": "홈 버튼 누르기",
            "arguments": {
                "command_type": "keyevent",
                "keycode": "KEYCODE_HOME"
            }
        },
        {
            "name": "설정 앱 실행",
            "arguments": {
                "command_type": "start_app",
                "package_name": "com.android.settings"
            }
        },
        {
            "name": "배터리 정보 조회",
            "arguments": {
                "command_type": "get_info",
                "shell_command": "dumpsys battery"
            }
        },
        {
            "name": "Chrome 패키지 검색",
            "arguments": {
                "command_type": "custom_shell",
                "shell_command": "pm list packages | grep chrome"
            }
        }
    ]
    
    for i, test_cmd in enumerate(test_commands, 1):
        print(f"\n{i}. {test_cmd['name']}...")
        result = controller.control_android_device(test_cmd['arguments'])
        
        if result.get('success'):
            print(f"   ✅ 성공")
            if result.get('result'):
                # 결과가 너무 길면 잘라서 표시
                output = result['result']
                if len(output) > 200:
                    output = output[:200] + "..."
                print(f"   결과: {output}")
        else:
            print(f"   ❌ 실패: {result.get('error', 'Unknown error')}")

def interactive_test():
    """대화형 테스트"""
    print("\n🔧 대화형 테스트")
    print("=" * 50)
    print("사용자가 직접 명령어를 입력하여 테스트할 수 있습니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print()
    
    controller = AndroidDeviceController()
    
    while True:
        try:
            command = input("명령어 입력 (예: tap 500 1000, keyevent KEYCODE_HOME): ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("테스트를 종료합니다.")
                break
            
            if not command:
                continue
            
            # 간단한 명령어 파싱
            parts = command.split()
            if len(parts) == 0:
                continue
            
            # 명령어 타입별 처리
            if parts[0] == 'tap' and len(parts) == 3:
                x, y = int(parts[1]), int(parts[2])
                result = controller.tap(x, y)
                print(f"터치 결과: {result}")
                
            elif parts[0] == 'keyevent' and len(parts) == 2:
                keycode = parts[1]
                result = controller.keyevent(keycode)
                print(f"키 이벤트 결과: {result}")
                
            elif parts[0] == 'text' and len(parts) >= 2:
                text = ' '.join(parts[1:])
                result = controller.input_text(text)
                print(f"텍스트 입력 결과: {result}")
                
            elif parts[0] == 'info':
                result = controller.get_device_info()
                print(f"디바이스 정보: {result}")
                
            elif parts[0] == 'packages':
                result = controller.list_packages()
                print(f"패키지 목록: {result}")
                
            else:
                print("지원되는 명령어:")
                print("  tap <x> <y> - 화면 터치")
                print("  keyevent <keycode> - 키 이벤트")
                print("  text <message> - 텍스트 입력")
                print("  info - 디바이스 정보")
                print("  packages - 패키지 목록")
                
        except KeyboardInterrupt:
            print("\n테스트를 종료합니다.")
            break
        except Exception as e:
            print(f"오류: {e}")

def main():
    """메인 테스트 함수"""
    print("🤖 AndroidDeviceController 테스트 스크립트")
    print("=" * 60)
    
    # 기본 기능 테스트
    if not test_basic_functionality():
        print("\n❌ 기본 기능 테스트 실패. ADB 설치 및 디바이스 연결을 확인하세요.")
        return
    
    # 개별 메서드 테스트
    test_individual_methods()
    
    # Tool-calling 통합 테스트
    test_tool_calling_integration()
    
    # 대화형 테스트 (선택사항)
    print("\n대화형 테스트를 시작하시겠습니까? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes', '예']:
            interactive_test()
    except KeyboardInterrupt:
        pass
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main() 