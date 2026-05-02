import itertools
import torch
import torch.nn.functional as F


def nll(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean NLL (cross-entropy) over all samples."""
    return F.nll_loss(probs.clamp(min=1e-7).log(), labels)


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean Brier score over all samples."""
    one_hot = F.one_hot(labels, num_classes=probs.shape[-1]).float()
    return ((probs - one_hot) ** 2).sum(dim=-1).mean()


def _sort_by_confidence(
    accuracies: torch.Tensor, confidences: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.argsort(confidences, descending=True, stable=True)
    return accuracies[idx], confidences[idx]


def _risk_coverage_curve(
    accuracies: torch.Tensor, confidences: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    acc_s, conf_s = _sort_by_confidence(accuracies, confidences)
    n = len(acc_s)
    counts = torch.arange(1, n + 1, dtype=acc_s.dtype, device=acc_s.device)
    coverages = counts / n
    risks = 1 - torch.cumsum(acc_s, dim=0) / counts
    return risks, coverages, conf_s


def auc_risk_coverage(accuracies: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
    """Area under the risk-coverage curve × 100 (lower is better)."""
    risks, coverages, _ = _risk_coverage_curve(accuracies, confidences)
    return 100 * torch.trapezoid(risks, coverages)


def coverage_at_risk(
    accuracies: torch.Tensor,
    confidences: torch.Tensor,
    risk_levels: list[float] = (0.05, 0.1, 0.2),
) -> dict[str, torch.Tensor]:
    """Coverage (× 100) achieved at each specified risk level."""
    risks, coverages, conf_s = _risk_coverage_curve(accuracies, confidences)
    # Keep only the last position within each run of equal confidences
    keep = torch.cat([
        torch.diff(conf_s) < -1e-6,
        torch.tensor([True], dtype=torch.bool, device=conf_s.device),
    ])
    risks_f, coverages_f = risks[keep], coverages[keep]

    def _find(risk_level: float) -> torch.Tensor:
        below = risks_f < risk_level
        if not below.any():
            return torch.tensor(0.0, device=accuracies.device)
        return 100 * coverages_f[below.nonzero()[-1].item()]

    return {f"C@{r}": _find(r) for r in risk_levels}


def expected_calibration_error(
    accuracies: torch.Tensor,
    confidences: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """ECE with equal-width bins × 100."""
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=confidences.device)

    def _bin_error(lo, hi):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            return torch.tensor(0.0, device=confidences.device)
        return mask.float().mean() * (accuracies[mask].mean() - confidences[mask].mean()).abs()

    return 100 * sum(_bin_error(lo, hi) for lo, hi in itertools.pairwise(bin_edges))


def adaptive_calibration_error(
    accuracies: torch.Tensor,
    confidences: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """ACE with equal-count bins × 100."""
    conf_s, idx = torch.sort(confidences, stable=True)
    acc_s = accuracies[idx]
    boundaries = torch.linspace(0, len(conf_s), n_bins + 1).long()

    def _bin_error(lo, hi):
        if hi <= lo:
            return torch.tensor(0.0, device=confidences.device)
        weight = (hi - lo) / len(accuracies)
        return weight * (acc_s[lo:hi].mean() - conf_s[lo:hi].mean()).abs()

    return 100 * sum(
        _bin_error(int(boundaries[i]), int(boundaries[i + 1]))
        for i in range(n_bins)
    )


def compute_all_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    risk_levels: list[float] = (0.05, 0.1, 0.2),
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute all metrics from MC-averaged softmax probabilities and integer labels."""
    confidences = probs.max(dim=-1).values
    accuracies = (probs.argmax(dim=-1) == labels).float()

    results = {
        "nll":   nll(probs, labels).item(),
        "brier": brier_score(probs, labels).item(),
        "ece":   expected_calibration_error(accuracies, confidences, n_bins).item(),
        "ace":   adaptive_calibration_error(accuracies, confidences, n_bins).item(),
        "auc_rc": auc_risk_coverage(accuracies, confidences).item(),
        **{k: v.item() for k, v in coverage_at_risk(accuracies, confidences, risk_levels).items()},
    }
    return results
