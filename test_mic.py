"""
Microphone Test Script
Test if your microphone is working correctly
"""
import sys
import time
import numpy as np
import sounddevice as sd
from loguru import logger


def list_audio_devices():
    """List all audio devices"""
    print("\n📢 Available Audio Devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        input_ch = device['max_input_channels']
        output_ch = device['max_output_channels']
        device_type = []
        if input_ch > 0:
            device_type.append("🎤 INPUT")
        if output_ch > 0:
            device_type.append("🔊 OUTPUT")
        
        print(f"[{i}] {device['name']}")
        print(f"    Type: {' | '.join(device_type)}")
        if input_ch > 0:
            print(f"    Input channels: {input_ch}, Default SR: {device['default_samplerate']}")
        print()
    
    default_input = sd.query_devices(kind='input')
    print(f"✅ Default Input Device: {default_input['name']}")
    print("-" * 60)


def test_microphone(duration: int = 5, sample_rate: int = 48000):
    """
    Test microphone capture
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate to use
    """
    print(f"\n🎤 Testing microphone for {duration} seconds...")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Speak into your microphone!\n")
    
    # Record audio
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        
        # Show live levels
        print("📊 Live audio levels:")
        for i in range(duration * 10):
            time.sleep(0.1)
            # Get current audio level
            current_sample = int(i * sample_rate / 10)
            end_sample = min(current_sample + int(sample_rate / 10), len(recording))
            if current_sample < len(recording):
                chunk = recording[current_sample:end_sample]
                if len(chunk) > 0:
                    rms = np.sqrt(np.mean(chunk**2))
                    db = 20 * np.log10(max(rms, 1e-10))
                    
                    # Visual bar
                    bar_len = int(max(0, min(50, (db + 60) / 60 * 50)))
                    bar = "█" * bar_len + "░" * (50 - bar_len)
                    print(f"\r  [{bar}] {db:.1f} dB", end="", flush=True)
        
        sd.wait()
        print("\n")
        
        # Analyze recording
        rms = np.sqrt(np.mean(recording**2))
        peak = np.max(np.abs(recording))
        db_rms = 20 * np.log10(max(rms, 1e-10))
        db_peak = 20 * np.log10(max(peak, 1e-10))
        
        print("📈 Recording Analysis:")
        print(f"   RMS Level: {db_rms:.1f} dB")
        print(f"   Peak Level: {db_peak:.1f} dB")
        
        if db_rms < -50:
            print("\n⚠️  WARNING: Audio level very low!")
            print("   - Check if microphone is muted")
            print("   - Try speaking louder or closer to mic")
            print("   - Check Windows microphone settings")
        elif db_rms < -35:
            print("\n⚠️  Audio level is low but detectable")
            print("   - Try speaking louder")
        else:
            print("\n✅ Audio level is good!")
        
        # Ask to playback
        response = input("\n🔊 Play back recording? (y/n): ")
        if response.lower() == 'y':
            print("Playing...")
            sd.play(recording, sample_rate)
            sd.wait()
            print("Done!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("🎤 Microphone Test for Voice-to-Voice System")
    print("="*60)
    
    # List devices
    list_audio_devices()
    
    # Test at different sample rates
    print("\n" + "-"*60)
    print("Testing at 48000 Hz (Ultravox default)...")
    test_microphone(duration=5, sample_rate=48000)
    
    print("\n" + "-"*60)
    print("✅ Microphone test complete!")
    print("\nIf your mic works here but not in Ultravox:")
    print("1. Check browser/app microphone permissions")
    print("2. Make sure no other app is using the microphone")
    print("3. Try setting a different default microphone in Windows")


if __name__ == "__main__":
    main()
