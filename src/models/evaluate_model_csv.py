#!/usr/bin/env python3
"""
Tennessee Eastman Process CSV 데이터 평가 스크립트
학습된 GAN v5 모델을 사용하여 CSV 테스트 데이터 평가
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import click
import logging
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.dataset import TEP_MEAN, TEP_STD, CSVToTensor, CSVNormalize, TEPCSVDataset


def setup_logger():
    """로거 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model(model_path, device):
    """학습된 모델 로드"""
    logger = logging.getLogger(__name__)
    logger.info(f"모델 로드 중: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 전체 모델 로드
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    logger.info(f"모델 로드 완료: {type(model).__name__}")
    return model


def evaluate_model(model, test_loader, device, logger):
    """모델 평가 수행"""
    logger.info("모델 평가 시작...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data["shot"].to(device)
            labels = data["label"].to(device)
            
            # 입력 형태 조정
            inputs = inputs.squeeze(dim=1)  # [batch, seq_len, features]
            labels = labels.squeeze()       # [batch, seq_len]
            
            # 모델 예측
            type_logits, real_fake_logits = model(inputs, None)
            
            # 확률 계산
            probabilities = torch.softmax(type_logits, dim=-1)
            
            # 예측 값 계산
            predictions = torch.argmax(type_logits, dim=-1)
            
            # 배치 데이터 저장
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if batch_idx % 10 == 0:
                logger.info(f"평가 진행률: {batch_idx + 1}/{len(test_loader)}")
    
    logger.info("모델 평가 완료")
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def analyze_results(predictions, labels, logger):
    """Run-level result analysis and metric calculation"""
    logger.info("런 단위 결과 분석 시작...")
    
    run_predictions = []
    run_labels = []
    
    # 각 시뮬레이션 런의 예측 결과를 다수결로 결정
    for i in range(len(predictions)):
        # 각 런의 예측 결과 (500개 시점)
        run_pred = predictions[i]
        run_label = labels[i][0]  # 런의 실제 라벨 (모든 시점 동일)
        
        # 다수결로 런의 예측 라벨 결정
        unique, counts = np.unique(run_pred, return_counts=True)
        majority_pred = unique[np.argmax(counts)]
        
        run_predictions.append(majority_pred)
        run_labels.append(run_label)
    
    run_predictions = np.array(run_predictions)
    run_labels = np.array(run_labels)
    
    # 런 단위 정확도
    run_accuracy = accuracy_score(run_labels, run_predictions)
    total_correct = np.sum(run_predictions == run_labels)
    total_runs = len(run_labels)
    
    logger.info(f"\n=== 시뮬레이션 런 단위 평가 결과 ===")
    logger.info(f"총 정답: {total_correct}/{total_runs} = {run_accuracy:.4f}")
    
    # 실제 사용하는 클래스 확인 (0~12)
    unique_classes = np.unique(np.concatenate([run_labels, run_predictions]))
    max_class = max(unique_classes)
    num_classes = min(13, max_class + 1)  # 최대 13개 클래스 (0~12)
    
    # 런 단위 클래스별 성능
    run_precision, run_recall, run_f1, run_support = precision_recall_fscore_support(
        run_labels, run_predictions, average=None, zero_division=0
    )
    
    logger.info("\n=== 각 결함별 정답률 ===")
    correct_per_class = []
    for i in range(num_classes):  # 실제 사용하는 클래스만 출력
        fault_type = "정상" if i == 0 else f"결함{i}"
        correct_count = np.sum((run_labels == i) & (run_predictions == i))
        total_count = np.sum(run_labels == i)
        correct_per_class.append(correct_count)
        accuracy_rate = correct_count/total_count if total_count > 0 else 0
        logger.info(f"{fault_type:>6}: {correct_count:3d}/{total_count:3d} = {accuracy_rate:.3f}")
    
    # 전체 평균 성능 (실제 사용하는 클래스만)
    avg_precision = np.mean(run_precision[:num_classes])
    avg_recall = np.mean(run_recall[:num_classes])
    avg_f1 = np.mean(run_f1[:num_classes])
    
    logger.info(f"\n=== 전체 평균 성능 ===")
    logger.info(f"평균 정밀도: {avg_precision:.4f}")
    logger.info(f"평균 재현율: {avg_recall:.4f}")
    logger.info(f"평균 F1 점수: {avg_f1:.4f}")
    
    return {
        'run_accuracy': run_accuracy,
        'run_predictions': run_predictions,
        'run_labels': run_labels,
        'run_precision': run_precision[:num_classes],
        'run_recall': run_recall[:num_classes],
        'run_f1': run_f1[:num_classes],
        'run_support': run_support[:num_classes],
        'num_classes': num_classes,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'total_correct': total_correct,
        'total_runs': total_runs
    }


def plot_confusion_matrix(run_predictions, run_labels, save_path=None):
    """Run-level confusion matrix visualization"""
    logger = logging.getLogger(__name__)
    logger.info("런 단위 혼동 행렬 생성 중...")
    
    # 실제 사용하는 클래스 확인
    unique_classes = np.unique(np.concatenate([run_labels, run_predictions]))
    max_class = max(unique_classes)
    num_classes = min(13, max_class + 1)  # 최대 13개 클래스 (0~12)
    
    # 혼동 행렬 계산 (런 단위)
    cm = confusion_matrix(run_labels, run_predictions)
    
    # 시각화
    plt.figure(figsize=(12, 10))
    class_names = ["Normal"] + [f"Fault{i}" for i in range(1, num_classes)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Tennessee Eastman Process Run-Level Fault Classification Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"혼동 행렬 저장: {save_path}")
    
    plt.show()


def save_results(results, save_dir):
    """Save run-level results"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save metrics
    metrics_file = os.path.join(save_dir, 'evaluation_metrics_run_level.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("Tennessee Eastman Process 런 단위 결함 탐지 평가 결과\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"총 정답: {results['total_correct']}/{results['total_runs']} = {results['run_accuracy']:.4f}\n\n")
        
        f.write("각 결함별 정답률:\n")
        for i in range(results['num_classes']):
            fault_type = "정상" if i == 0 else f"결함{i}"
            correct_count = np.sum((results['run_labels'] == i) & (results['run_predictions'] == i))
            total_count = np.sum(results['run_labels'] == i)
            accuracy_rate = correct_count/total_count if total_count > 0 else 0
            f.write(f"  {fault_type}: {correct_count:3d}/{total_count:3d} = {accuracy_rate:.3f}\n")
        
        f.write(f"\n전체 평균 성능:\n")
        f.write(f"  평균 정밀도: {results['avg_precision']:.4f}\n")
        f.write(f"  평균 재현율: {results['avg_recall']:.4f}\n")
        f.write(f"  평균 F1 점수: {results['avg_f1']:.4f}\n")
    
    # 런 단위 예측 결과 저장
    pred_file = os.path.join(save_dir, 'run_predictions.npz')
    np.savez(pred_file, 
             run_predictions=results['run_predictions'], 
             run_labels=results['run_labels'])
    
    # 런 단위 혼동 행렬 저장
    cm_file = os.path.join(save_dir, 'confusion_matrix_run_level.png')
    plot_confusion_matrix(results['run_predictions'], results['run_labels'], cm_file)
    
    logger.info(f"런 단위 평가 결과 저장 완료: {save_dir}")


@click.command()
@click.option('--model_path', required=True, type=str, help='학습된 discriminator 모델 경로 (.pth 파일)')
@click.option('--csv_dir', type=str, default='data/test_faults', help='CSV 파일들이 있는 디렉토리')
@click.option('--cuda', type=int, default=0, help='사용할 GPU 번호')
@click.option('--batch_size', type=int, default=16, help='배치 크기')
@click.option('--save_dir', type=str, default='evaluation_results_csv', help='결과 저장 디렉토리')
@click.option('--random_seed', type=int, default=42, help='랜덤 시드')
def main(model_path, csv_dir, cuda, batch_size, save_dir, random_seed):
    """
    Tennessee Eastman Process CSV 데이터 평가
    
    사용법:
    python src/models/evaluate_model_csv.py --model_path models/4_main_model/weights/199_epoch_discriminator.pth --cuda 0
    """
    
    # 로거 설정
    logger = setup_logger()
    logger.info("Tennessee Eastman Process CSV 데이터 평가 시작")
    
    # 랜덤 시드 설정
    logger.info(f"Random Seed: {random_seed}")
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 디바이스 설정
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")
    
    # CSV 파일 목록 생성
    csv_files = [os.path.join(csv_dir, f"test_fault_{i}.csv") for i in range(13)]
    
    # 존재하는 파일만 필터링
    existing_files = [f for f in csv_files if os.path.exists(f)]
    logger.info(f"발견된 CSV 파일: {len(existing_files)}개")
    
    if not existing_files:
        logger.error("CSV 파일을 찾을 수 없습니다!")
        return
    
    # 데이터 변환 설정
    transform = transforms.Compose([
        CSVToTensor(),
        CSVNormalize()
    ])
    
    # 테스트 데이터셋 로드
    logger.info("CSV 데이터셋 로드 중...")
    test_dataset = TEPCSVDataset(existing_files, transform=transform, is_test=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 멀티프로세싱 오류 방지
        drop_last=False
    )
    
    logger.info(f"테스트 데이터셋 크기: {len(test_dataset)}")
    
    # 모델 로드
    model = load_model(model_path, device)
    
    # 모델 평가
    predictions, labels, probabilities = evaluate_model(model, test_loader, device, logger)
    
    # 결과 분석
    results = analyze_results(predictions, labels, logger)
    
    # 결과 저장
    save_results(results, save_dir)
    
    logger.info("CSV 데이터 평가 완료!")


if __name__ == '__main__':
    main() 