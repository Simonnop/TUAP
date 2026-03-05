from attack.base_attack import BaseAttack
from attack.global_attack import BaseGlobalAttack, GlobalGradientAccumulationAttack
from attack.global_timestamp_wise_fix import GlobalTimestampWiseFix
from attack.global_ablation import Ablation
from attack.empty_global_attack import EmptyGlobalAttack
from attack.direction_attack import DirectionAttack
from attack.tca_attack import TCAAttack
from attack.more.nifgsm import NIFGSM, GGAA_NIFGSM
from attack.more.vmifgsm import VMIFGSM, GGAA_VMIFGSM
from attack.more.fgsm import FGSM, GGAA_FGSM
from attack.more.bim import BIM, GGAA_BIM
from attack.more.pgd import PGD, GGAA_PGD
from attack.more.iefgsm import IEFGSM, GGAA_IEFGSM
from attack.more.gifgsm import GIFGSM, GGAA_GIFGSM
from attack.more.pifgsm import PIFGSM, GGAA_PIFGSM
from attack.more.empty import EMPTY


class Attacker:
    def __init__(self, args, generate_model, metrics, device):
        self.attack_algo = args.attack_algo
        self.generate_model = generate_model
        # self.norm = args.norm
        self.norm = "linfty"
        self.time_window = args.seq_len
        self.device = device
        self.metrics = metrics
        self.args = args

    def get_attacker(
        self,
        epsilon,
        global_=False,
        aug_kind=None,
        by_direction=False,
        attack_algo=None,
    ):
        # 从 args 中获取 epoch 和 alpha_times 参数
        epoch = self.args.epoch
        alpha_times = self.args.alpha_times
        if attack_algo is not None:
            self.attack_algo = attack_algo

        if by_direction:
            attacker_clazz = DirectionAttack
            attacker = attacker_clazz(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                alpha_times=alpha_times,
            )
        elif global_:
            # 丢弃比例
            drop_ratio = 0.5 if aug_kind == "DROPOUT" else 0
            # 攻击器
            if aug_kind in (None, "DROPOUT"):
                if self.attack_algo == "GGAA_NIFGSM":
                    attacker_clazz = GGAA_NIFGSM
                elif self.attack_algo == "GGAA_VMIFGSM":
                    attacker_clazz = GGAA_VMIFGSM
                elif self.attack_algo == "GGAA_FGSM":
                    attacker_clazz = GGAA_FGSM
                elif self.attack_algo == "GGAA_BIM":
                    attacker_clazz = GGAA_BIM
                elif self.attack_algo == "GGAA_PGD":
                    attacker_clazz = GGAA_PGD
                elif self.attack_algo == "GGAA_IEFGSM":
                    attacker_clazz = GGAA_IEFGSM
                elif self.attack_algo == "GGAA_GIFGSM":
                    attacker_clazz = GGAA_GIFGSM
                elif self.attack_algo == "GGAA_PIFGSM":
                    attacker_clazz = GGAA_PIFGSM
                elif self.attack_algo in ["GGAA"]:
                    attacker_clazz = GlobalGradientAccumulationAttack
                elif self.attack_algo == "GGAA_First":
                    attacker_clazz = Ablation
                elif self.attack_algo == "GGAA_Last":
                    attacker_clazz = Ablation
                elif self.attack_algo == "GGAA_Random":
                    attacker_clazz = Ablation
                elif self.attack_algo == "GTW_Fix":
                    attacker_clazz = GlobalTimestampWiseFix
                elif self.attack_algo == "EMPTY_GLOBAL":
                    attacker_clazz = EmptyGlobalAttack
                else:
                    attacker_clazz = BaseGlobalAttack
            else:
                raise ValueError(f"Unsupported aug kind: {aug_kind}")
            decay = getattr(self.args, 'mu', 1.0)
            # Ablation 需要 ablation_type 参数
            if attacker_clazz is Ablation:
                ablation_type = 'first' if self.attack_algo == "GGAA_First" else 'last' if self.attack_algo == "GGAA_Last" else 'random'
                attacker = attacker_clazz(
                    self.attack_algo,
                    self.generate_model,
                    epsilon,
                    norm=self.norm,
                    metrics=self.metrics,
                    device=self.device,
                    args=self.args,
                    time_window=self.time_window,
                    # drop_ratio=drop_ratio,
                    epoch=epoch,
                    alpha_times=alpha_times,
                    decay=decay,
                    ablation_type=ablation_type,
                )
            else:
                kwargs = dict(
                    attack=self.attack_algo,
                    model=self.generate_model,
                    epsilon=epsilon,
                    norm=self.norm,
                    metrics=self.metrics,
                    device=self.device,
                    args=self.args,
                    time_window=self.time_window,
                    # drop_ratio=drop_ratio,
                    epoch=epoch,
                    alpha_times=alpha_times,
                    decay=decay,
                )
                if attacker_clazz in (GGAA_GIFGSM,):
                    kwargs['pre_epoch'] = getattr(self.args, 'pre_epoch', 5)
                    kwargs['s'] = getattr(self.args, 's', 10)
                elif attacker_clazz in (GGAA_PIFGSM,):
                    kwargs['decay'] = 0.
                    kwargs['kern_size'] = getattr(self.args, 'kern_size', 3)
                    kwargs['gamma'] = getattr(self.args, 'gamma', 16.0)
                    kwargs['beta'] = getattr(self.args, 'beta', 10.0)
                attacker = attacker_clazz(**kwargs)
        elif self.attack_algo == "TCA":
            attacker = TCAAttack(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                alpha_times=alpha_times,
                lambda_mean=getattr(self.args, "lambda_mean", 0.1),
                lambda_std=getattr(self.args, "lambda_std", 0.1),
                lambda_trend=getattr(self.args, "lambda_trend", 0.1),
            )
        elif self.attack_algo == "NIFGSM":
            attacker = NIFGSM(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                metrics=self.metrics,
                alpha_times=alpha_times,
            )
        elif self.attack_algo == "VMIFGSM":
            attacker = VMIFGSM(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                metrics=self.metrics,
                alpha_times=alpha_times,
            )
        elif self.attack_algo == "FGSM":
            attacker = FGSM(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                metrics=self.metrics,
                alpha_times=alpha_times,
            )
        elif self.attack_algo == "EMPTY":
            attacker = EMPTY(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=1,
                metrics=self.metrics,
                alpha_times=alpha_times,
            )
        elif self.attack_algo == "BIM":
            attacker = BIM(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                metrics=self.metrics,
                alpha_times=alpha_times,
            )
        elif self.attack_algo == "PGD":
            attacker = PGD(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                metrics=self.metrics,
                alpha_times=alpha_times,
            )
        elif self.attack_algo == "IEFGSM":
            attacker = IEFGSM(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                metrics=self.metrics,
                alpha_times=alpha_times,
            )
        elif self.attack_algo == "GIFGSM":
            attacker = GIFGSM(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                metrics=self.metrics,
                alpha_times=alpha_times,
                pre_epoch=getattr(self.args, 'pre_epoch', 5),
                s=getattr(self.args, 's', 10),
            )
        elif self.attack_algo == "PIFGSM":
            attacker = PIFGSM(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                decay=0.,
                metrics=self.metrics,
                alpha_times=alpha_times,
                kern_size=getattr(self.args, 'kern_size', 3),
                gamma=getattr(self.args, 'gamma', 16.0),
                beta=getattr(self.args, 'beta', 10.0),
            )
        else:
            decay = getattr(self.args, 'mu', 1.0)
            attacker = BaseAttack(
                self.attack_algo,
                self.generate_model,
                epsilon,
                norm=self.norm,
                device=self.device,
                args=self.args,
                epoch=epoch,
                metrics=self.metrics,
                alpha_times=alpha_times,
                decay=decay,
            )
        self.attacker = attacker
        return attacker
