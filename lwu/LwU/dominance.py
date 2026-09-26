"""Dominance-based zones with labeled-retain repair."""

import torch.nn as nn

from LwU.teacher_repair import TeacherRepair


class DominanceZones(TeacherRepair):
    """Teacher repair with dominance zones and CE-augmented Zone-A repair.

    A and B require relative retain/forget dominance.  Salient parameters
    without clear dominance enter C, and only low-both parameters enter D.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        importance: str = 'ssv',
        zone_c: str = 'granular',
        retain_quantile: float = 0.9,
        forget_quantile: float = 0.9,
        distill_epochs: int = 3,
        distill_lr: float = 1e-3,
        temperature: float = 4.0,
        ce_weight: float = 1.0,
        zone_rule: str = 'dominance',
        dominance_rule: str = 'scale_free',
        dominance_margin: float = 1.0,
        per_instance_forget_fisher: bool = False,
        device: str = 'cuda',
    ):
        if ce_weight <= 0.0:
            raise ValueError('DominanceZones requires ce_weight > 0')
        if zone_rule != 'dominance':
            raise ValueError("DominanceZones zone_rule must be 'dominance'")
        if dominance_rule not in ('scale_free', 'legacy_omega'):
            raise ValueError("dominance_rule must be 'scale_free' or 'legacy_omega'")
        if dominance_margin < 1.0:
            raise ValueError('dominance_margin must be at least 1')
        self.zone_rule = zone_rule
        self.dominance_rule = dominance_rule
        self.dominance_margin = float(dominance_margin)
        self.dominance_omega = 1.0
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            importance=importance,
            zone_c=zone_c,
            retain_quantile=retain_quantile,
            forget_quantile=forget_quantile,
            distill_epochs=distill_epochs,
            distill_lr=distill_lr,
            temperature=temperature,
            ce_weight=ce_weight,
            per_instance_forget_fisher=per_instance_forget_fisher,
            device=device,
        )

    def identify_zones(self, retain_loader, forget_loader=None, criterion=None):
        """Build an exhaustive partition using relative importance dominance.

        Dominance compares two importance maps, so it has to be scale-free.
        ``dominance_rule='scale_free'`` (the default) divides each map by its own
        saliency threshold first and asks which map a parameter is more extreme
        in.  ``'legacy_omega'`` reproduces the original single-factor rule and
        exists only to regenerate results published before the fix.

        The legacy rule multiplied *both* directions by
        ``omega = max(tau_f/tau_r, tau_r/tau_f)``, conflating three unrelated
        quantities: the scale gap between the two Fisher estimates, the gap
        between the retain and forget quantiles, and the intended dominance
        margin.  Because omega is >= 1 by construction it penalised whichever
        map ran colder, so when the forget Fisher was estimated at batch size 1
        against a retain Fisher at batch size 64 -- a 187x scale gap, and the
        configuration the instance-deletion benchmark used -- retain dominance
        became unreachable and Zone A came out empty on every run.
        """
        super().identify_zones(retain_loader, forget_loader, criterion)
        tau_r = max(self.resolved_tau_r, 1e-30)
        tau_f = max(self.resolved_tau_f, 1e-30)
        self.dominance_omega = max(tau_f / tau_r, tau_r / tau_f)

        for name in self.phi_r:
            retain_salient = self.phi_r[name] > self.resolved_tau_r
            forget_salient = self.phi_f[name] > self.resolved_tau_f
            if self.dominance_rule == 'scale_free':
                # Each map in units of its own threshold, then a plain
                # comparison; the margin stays separate and explicit.
                retain_score = self.phi_r[name] / tau_r
                forget_score = self.phi_f[name] / tau_f
                margin = self.dominance_margin
            else:
                retain_score = self.phi_r[name]
                forget_score = self.phi_f[name]
                margin = self.dominance_omega
            retain_test = retain_score >= margin * forget_score
            forget_test = forget_score >= margin * retain_score

            # With margin=1, exact ties satisfy both >= tests.  A tie is not
            # dominance, so it belongs to C rather than overlapping A and B.
            retain_dominant = retain_test & ~forget_test
            forget_dominant = forget_test & ~retain_test
            salient = retain_salient | forget_salient

            zone_a = retain_salient & retain_dominant
            zone_b = forget_salient & forget_dominant
            self.zone_masks['A'][name] = zone_a
            self.zone_masks['B'][name] = zone_b
            self.zone_masks['C'][name] = salient & ~zone_a & ~zone_b
            self.zone_masks['D'][name] = ~salient

    def unlearn(self, *args, **kwargs):
        result = super().unlearn(*args, **kwargs)
        self.diagnostics['zone_construction'] = (
            'dominance: A=retain-dominant, B=forget-dominant, ' 'C=salient non-dominant, D=low-both'
        )
        self.diagnostics['dominance_omega'] = self.dominance_omega
        self.diagnostics['dominance_rule'] = self.dominance_rule
        self.diagnostics['dominance_margin'] = self.dominance_margin
        return result
