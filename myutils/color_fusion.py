import numpy as np
import cv2


##################
## EM Algorithm ##
##################


def initialize(A, B):
    F = (A + B) / 2
    alpha = np.zeros(A.shape + (2,), dtype=int)  # 마지막 차원: A, B
    beta = np.zeros(A.shape + (2,))
    return F, alpha, beta


def e_step(A, B, F, beta, sigma):
    # \alpha의 가능한 값: 1, 0, -1
    alphas = np.array([1, 0, -1])

    # Initialize responsibility arrays
    gamma_A = np.zeros(A.shape + (len(alphas),))
    gamma_B = np.zeros(B.shape + (len(alphas),))

    # Compute likelihoods for each possible alpha
    for idx, alpha_val in enumerate(alphas):
        # For image A
        L_A = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
            -((A - alpha_val * F - beta[..., 0]) ** 2) / (2 * sigma**2)
        )
        gamma_A[..., idx] = L_A

        # For image B
        L_B = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
            -((B - alpha_val * F - beta[..., 1]) ** 2) / (2 * sigma**2)
        )
        gamma_B[..., idx] = L_B

    # Sum over all possible alphas for normalization
    sum_gamma = (
        gamma_A.sum(axis=-1) + gamma_B.sum(axis=-1) + 1e-8
    )  # Avoid division by zero

    # Compute posterior probabilities
    gamma_A /= sum_gamma[..., np.newaxis]
    gamma_B /= sum_gamma[..., np.newaxis]

    return gamma_A, gamma_B


def m_step(A, B, gamma_A, gamma_B, alphas):
    # alphas: (3,) -> reshape to (1, 1, 3) for broadcasting with (540,720,3)
    alphas_reshaped = alphas[np.newaxis, np.newaxis, :]  # (1, 1, 3)

    # 확장된 A와 B
    A_expanded = A[..., np.newaxis]  # (540, 720, 1)
    B_expanded = B[..., np.newaxis]  # (540, 720, 1)

    # Update F
    numerator = (
        gamma_A * alphas_reshaped * A_expanded + gamma_B * alphas_reshaped * B_expanded
    )
    denominator = (gamma_A + gamma_B) * alphas_reshaped + 1e-8  # Avoid division by zero

    # Compute F as weighted average
    F_new = numerator.sum(axis=-1) / denominator.sum(axis=-1)

    # Update beta
    beta_A = (A_expanded - F_new[..., np.newaxis] * alphas_reshaped) * gamma_A
    beta_B = (B_expanded - F_new[..., np.newaxis] * alphas_reshaped) * gamma_B

    # Sum over alphas
    beta_A = beta_A.sum(axis=-1) / (gamma_A.sum(axis=-1) + 1e-8)
    beta_B = beta_B.sum(axis=-1) / (gamma_B.sum(axis=-1) + 1e-8)

    # Update alpha by selecting the alpha with the highest posterior probability
    alpha_A = alphas[np.argmax(gamma_A, axis=-1)]
    alpha_B = alphas[np.argmax(gamma_B, axis=-1)]

    return F_new, alpha_A, alpha_B, beta_A, beta_B


def em_algorithm(A, B, max_iters=100, tol=1e-3, sigma=1.0):
    F, alpha, beta = initialize(A, B)
    alphas = np.array([1, 0, -1])
    for iteration in range(max_iters):
        F_old = F.copy()
        # E 단계
        gamma_A, gamma_B = e_step(A, B, F, beta, sigma)

        # M 단계
        F, alpha_A, alpha_B, beta_A, beta_B = m_step(A, B, gamma_A, gamma_B, alphas)
        beta = np.stack([beta_A, beta_B], axis=-1)

        # 알파 업데이트
        alpha[..., 0] = alpha_A
        alpha[..., 1] = alpha_B

        # 수렴 검사
        delta = np.linalg.norm(F - F_old)
        if delta < tol:
            break
    return F, alpha, beta


def estimate_exposure_ratio(gray1, gray2):
    return np.log2(np.mean(gray2) / np.mean(gray1))


def compute_hdr(img1, img2):
    exposure_ratio = estimate_exposure_ratio(img1, img2)
    exposure_times = np.array([1.0, 2.0**exposure_ratio], dtype=np.float32)
    img1 = np.repeat(img1[..., np.newaxis] * 255, 3, axis=-1).astype(np.uint8)
    img2 = np.repeat(img2[..., np.newaxis] * 255, 3, axis=-1).astype(np.uint8)
    images = [img1, img2]
    merge_debevec = cv2.createMergeRobertson()
    hdr = merge_debevec.process(images, times=exposure_times)
    return hdr


def false_fusion_algorithm(rgb: np.ndarray, nir: np.ndarray):
    yuv_mat = np.array(
        [[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.100]]
    )
    # transform rgb to yuv
    yuv = np.dot(rgb, yuv_mat.T)
    y, u, v = yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]
    # brightness correction
    yh = compute_hdr(y, nir).mean(axis=-1)
    yh2 = (yh + y) / 2
    yh = (yh + nir) / 2
    u = u * yh / y
    v = v * yh / y
    # compute y channel using EM algorithm
    mul = 25
    y_em, alpha, beta = em_algorithm(yh * mul, yh2 * mul, max_iters=25)
    y_em /= mul

    # mix u and v channel with nir
    un = u - nir
    vn = nir - v
    y = yh

    # color correction by standard deviation
    def mix_std(x, y):
        return np.sqrt(np.var(x)) / np.sqrt(np.var(y)) * (y - np.mean(y)) + np.mean(x)

    yn = mix_std(y, y_em)
    un = mix_std(u, un)

    d = np.abs(nir - np.mean(nir))
    mu = d / np.mean(d)
    vn = mu * mix_std(v, vn)

    yuv = np.stack([yn, un, vn], axis=2)
    rgb_mat = np.array([[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]])
    rgb = np.dot(yuv, rgb_mat.T)
    return rgb, [yh2, y_em]
