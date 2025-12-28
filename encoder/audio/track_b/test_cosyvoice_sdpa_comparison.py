#!/usr/bin/env python3
# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

"""
SDPA 사용 여부에 따른 MultiHeadAttention 결과 비교 테스트
"""
import torch
import sys
import os

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.model.cosyvoice import MultiHeadAttention, AudioEncoderV2


def create_mask(batch_size, seq_len, device):
    """Attention mask 생성 (padding mask)"""
    # 예시: 각 시퀀스의 길이를 다르게 설정
    lengths = torch.tensor([seq_len] * batch_size, device=device)
    if batch_size >= 2:
        lengths[1] = max(1, seq_len - 2)
    if batch_size >= 3:
        lengths[2] = max(1, seq_len - 5)
    
    max_len = seq_len
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    mask = ~mask  # True for non-padded, False for padded
    
    # bias mask로 변환 (False -> -1e10, True -> 0)
    # mask shape: (B, T) -> (B, 1, T) -> (B, 1, 1, T) for broadcasting with (B, n_head, T, T)
    mask_bias = mask.to(torch.float32)
    mask_bias = (1.0 - mask_bias) * -1.0e10
    return mask_bias.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)


def test_qkv_attention_comparison():
    """qkv_attention 메서드 직접 비교"""
    print("=" * 80)
    print("qkv_attention 메서드 직접 비교 테스트")
    print("=" * 80)
    
    # 테스트 파라미터
    batch_size = 2
    seq_len = 10
    n_state = 1280
    n_head = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"n_state: {n_state}, n_head: {n_head}")
    print()
    
    # 동일한 입력 생성
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, n_state, device=device)
    k = torch.randn(batch_size, seq_len, n_state, device=device)
    v = torch.randn(batch_size, seq_len, n_state, device=device)
    mask = create_mask(batch_size, seq_len, device)
    
    # 두 개의 attention 모듈 생성 (가중치 공유)
    attn_no_sdpa = MultiHeadAttention(n_state, n_head, use_sdpa=False).to(device)
    attn_sdpa = MultiHeadAttention(n_state, n_head, use_sdpa=True).to(device)
    
    # 가중치 복사 (동일한 가중치 사용)
    attn_sdpa.load_state_dict(attn_no_sdpa.state_dict())
    
    # qkv_attention 직접 호출
    print("1. use_sdpa=False 경로 실행...")
    with torch.no_grad():
        output_no_sdpa, qk_no_sdpa = attn_no_sdpa.qkv_attention(q, k, v, mask)
    
    print("2. use_sdpa=True 경로 실행...")
    with torch.no_grad():
        output_sdpa, qk_sdpa = attn_sdpa.qkv_attention(q, k, v, mask)
    
    # 결과 비교
    print("\n" + "-" * 80)
    print("결과 비교:")
    print("-" * 80)
    print(f"Output shape (no_sdpa): {output_no_sdpa.shape}")
    print(f"Output shape (sdpa):    {output_sdpa.shape}")
    print()
    
    # 수치 비교
    max_diff = (output_no_sdpa - output_sdpa).abs().max().item()
    mean_diff = (output_no_sdpa - output_sdpa).abs().mean().item()
    rel_diff = ((output_no_sdpa - output_sdpa).abs() / (output_no_sdpa.abs() + 1e-8)).mean().item()
    
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Mean relative difference: {rel_diff:.2e}")
    print()
    
    # 거의 동일한지 확인 (부동소수점 오차 고려)
    tolerance = 1e-5
    is_close = torch.allclose(output_no_sdpa, output_sdpa, atol=tolerance, rtol=tolerance)
    print(f"Results are close (tol={tolerance}): {is_close}")
    
    if not is_close:
        print("\n⚠️  경고: 결과가 다릅니다!")
        # 차이가 큰 위치 찾기
        diff = (output_no_sdpa - output_sdpa).abs()
        max_idx = diff.argmax()
        print(f"최대 차이 위치: {max_idx}")
        print(f"no_sdpa 값: {output_no_sdpa.flatten()[max_idx].item():.6f}")
        print(f"sdpa 값:    {output_sdpa.flatten()[max_idx].item():.6f}")
    else:
        print("\n✅ 결과가 동일합니다!")
    
    return is_close


def test_full_attention_comparison():
    """전체 forward 메서드 비교"""
    print("\n" + "=" * 80)
    print("전체 forward 메서드 비교 테스트")
    print("=" * 80)
    
    # 테스트 파라미터
    batch_size = 2
    seq_len = 10
    n_state = 1280
    n_head = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"n_state: {n_state}, n_head: {n_head}")
    print()
    
    # 동일한 입력 생성
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, n_state, device=device)
    mask = create_mask(batch_size, seq_len, device)
    
    # 두 개의 attention 모듈 생성 (가중치 공유)
    attn_no_sdpa = MultiHeadAttention(n_state, n_head, use_sdpa=False).to(device)
    attn_sdpa = MultiHeadAttention(n_state, n_head, use_sdpa=True).to(device)
    
    # 가중치 복사
    attn_sdpa.load_state_dict(attn_no_sdpa.state_dict())
    
    # Forward pass
    print("1. use_sdpa=False 경로 실행...")
    with torch.no_grad():
        output_no_sdpa, qk_no_sdpa = attn_no_sdpa(x, mask)
    
    print("2. use_sdpa=True 경로 실행...")
    with torch.no_grad():
        output_sdpa, qk_sdpa = attn_sdpa(x, mask)
    
    # 결과 비교
    print("\n" + "-" * 80)
    print("결과 비교:")
    print("-" * 80)
    print(f"Output shape (no_sdpa): {output_no_sdpa.shape}")
    print(f"Output shape (sdpa):    {output_sdpa.shape}")
    print()
    
    max_diff = (output_no_sdpa - output_sdpa).abs().max().item()
    mean_diff = (output_no_sdpa - output_sdpa).abs().mean().item()
    rel_diff = ((output_no_sdpa - output_sdpa).abs() / (output_no_sdpa.abs() + 1e-8)).mean().item()
    
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Mean relative difference: {rel_diff:.2e}")
    print()
    
    tolerance = 1e-5
    is_close = torch.allclose(output_no_sdpa, output_sdpa, atol=tolerance, rtol=tolerance)
    print(f"Results are close (tol={tolerance}): {is_close}")
    
    if not is_close:
        print("\n⚠️  경고: 결과가 다릅니다!")
    else:
        print("\n✅ 결과가 동일합니다!")
    
    return is_close


def test_scale_verification():
    """Scale 적용 방식 검증"""
    print("\n" + "=" * 80)
    print("Scale 적용 방식 검증")
    print("=" * 80)
    
    batch_size = 1
    seq_len = 5
    n_state = 1280
    n_head = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    D = n_state
    scale = (D // n_head) ** -0.25
    print(f"Scale value: {scale:.6f}")
    print(f"Scale^2:     {scale**2:.6f}")
    print()
    
    # 간단한 예시로 수동 계산
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, n_state, device=device)
    k = torch.randn(batch_size, seq_len, n_state, device=device)
    v = torch.randn(batch_size, seq_len, n_state, device=device)
    
    # Reshape
    q_reshaped = q.view(batch_size, seq_len, n_head, -1)
    k_reshaped = k.view(batch_size, seq_len, n_head, -1)
    v_reshaped = v.view(batch_size, seq_len, n_head, -1)
    
    # no_sdpa 방식
    q_no_sdpa = q_reshaped.permute(0, 2, 1, 3) * scale  # (B, n_head, T, D//n_head)
    k_no_sdpa = k_reshaped.permute(0, 2, 3, 1) * scale  # (B, n_head, D//n_head, T)
    qk_no_sdpa = q_no_sdpa @ k_no_sdpa  # (B, n_head, T, T)
    
    # sdpa 방식
    q_sdpa = q_reshaped.permute(0, 2, 1, 3) * scale  # (B, n_head, T, D//n_head)
    k_sdpa = k_reshaped.permute(0, 2, 1, 3) * scale  # (B, n_head, T, D//n_head)
    # SDPA 내부: q @ k.transpose(-2, -1)
    qk_sdpa_manual = q_sdpa @ k_sdpa.transpose(-2, -1)  # (B, n_head, T, T)
    
    print("qk 계산 비교:")
    print(f"no_sdpa qk shape: {qk_no_sdpa.shape}")
    print(f"sdpa qk shape:    {qk_sdpa_manual.shape}")
    
    max_diff_qk = (qk_no_sdpa - qk_sdpa_manual).abs().max().item()
    print(f"qk max difference: {max_diff_qk:.2e}")
    
    if max_diff_qk < 1e-5:
        print("✅ qk 계산이 동일합니다!")
    else:
        print("⚠️  qk 계산이 다릅니다!")


if __name__ == "__main__":
    print("SDPA vs Non-SDPA 비교 테스트 시작\n")
    
    try:
        # Scale 검증
        test_scale_verification()
        
        # qkv_attention 직접 비교
        result1 = test_qkv_attention_comparison()
        
        # 전체 forward 비교
        result2 = test_full_attention_comparison()
        
        print("\n" + "=" * 80)
        print("전체 테스트 결과")
        print("=" * 80)
        print(f"qkv_attention 비교: {'✅ 통과' if result1 else '❌ 실패'}")
        print(f"forward 비교:       {'✅ 통과' if result2 else '❌ 실패'}")
        
        if result1 and result2:
            print("\n🎉 모든 테스트 통과!")
        else:
            print("\n⚠️  일부 테스트 실패 - 로직을 다시 확인해주세요.")
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

