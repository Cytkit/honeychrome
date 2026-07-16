from flowkit import transforms as Transforms
import numpy as np
import honeychrome.settings as settings

transforms_menu_items = ['Linear', 'Logicle', 'Log']

class Transform:
    def __init__(self, scale_t=262144, linear_a=100, logicle_w=0.5, logicle_m=4.5, logicle_a=0, log_m=6):
        self.xform = None
        self.scale = None
        self.step_scale = None
        self.zero_inverse = None
        self.zero = None
        self.ticks = None
        self.scale_t = scale_t
        self.linear_a = linear_a
        self.logicle_w = logicle_w
        self.logicle_m = logicle_m
        self.logicle_a = logicle_a
        self.log_m = log_m
        self.scale_bins = settings.hist_bins_retrieved
        self.limits = [0, 1]
        self.id = None

    def set_transform(self, id=None, limits=None):
        if limits is not None:
            self.limits = limits
        if id is not None:
            self.id = id

        if self.id == 0: #'linear'
            self.set_linear()
        elif self.id == 1: #'logicle'
            self.set_logicle()
        elif self.id == 2: #'log'
            self.set_log()
        else: #self.id == 'default':
            # Per-sample Time axis: use the limits passed in (the controller sets
            # these to each loaded sample's own observed range in
            # initialise_data_for_cytometry_plots). Previously the upper limit was
            # forced up to settings.default_ceiling, which squashed short runs and
            # was the "temporary fix" for time gates across samples of different
            # lengths — now the axis fits each sample.
            self.set_default()

    def set_linear(self):
        self.xform = Transforms.LinearTransform(param_t=self.scale_t, param_a=self.linear_a)
        limits = self.limits
        self.scale = np.concatenate((
            [-np.inf],
            self.xform.inverse(np.linspace(limits[0], limits[1], self.scale_bins)),
            [np.inf]
        ))
        self.step_scale = np.concatenate((
            [limits[0]-1/self.scale_bins],
            np.linspace(limits[0], limits[1], self.scale_bins),
            [limits[1]+1/self.scale_bins]
        ))
        self.zero_inverse = self.xform.inverse(np.array([0]))[0]
        self.zero = self.xform.apply(np.array([0]))[0]
        self.ticks = self.linear_ticks

    def set_logicle(self):
        self.xform = Transforms.LogicleTransform(param_t=self.scale_t, param_w=self.logicle_w, param_m=self.logicle_m, param_a=self.logicle_a)
        limits = self.limits
        self.scale = np.concatenate((
            [-np.inf],
            self.xform.inverse(np.linspace(limits[0], limits[1], self.scale_bins)),
            [np.inf]
        ))
        self.step_scale = np.concatenate((
            [limits[0]-1/self.scale_bins],
            np.linspace(limits[0], limits[1], self.scale_bins),
            [limits[1]+1/self.scale_bins]
        ))
        self.zero_inverse = self.xform.inverse(np.array([0]))[0]
        self.zero = self.xform.apply(np.array([0]))[0]
        self.ticks = self.logicle_ticks

    def set_log(self):
        self.xform = Transforms.LogTransform(param_t=self.scale_t, param_m=self.log_m)
        limits = self.limits
        self.scale = np.concatenate((
            [-np.inf],
            self.xform.inverse(np.linspace(limits[0], limits[1], self.scale_bins)),
            [np.inf]
        ))
        self.step_scale = np.concatenate((
            [limits[0]-1/self.scale_bins],
            np.linspace(limits[0], limits[1], self.scale_bins),
            [limits[1]+1/self.scale_bins]
        ))
        self.zero_inverse = -np.inf
        self.zero = self.xform.apply(np.array([1]))[0]
        self.ticks = self.log_ticks

    def set_default(self):
        self.xform = None
        limits = self.limits
        range = limits[1] - limits[0] + 1
        # Cap the bin count: Time can span tens of thousands of units, but a
        # per-unit grid makes a Time gate's 2D lookup table (bins_time x bins_other)
        # tens of millions of cells — slow to rebuild per sample. A few thousand
        # bins is plenty for both the Time histogram and gating.
        self.scale_bins = min(int(range), settings.default_time_max_bins)
        self.scale = np.concatenate((
            [-np.inf],
            np.linspace(limits[0], limits[1], self.scale_bins),
            [np.inf]
        ))
        self.step_scale = np.concatenate((
            [limits[0]-range/self.scale_bins],
            np.linspace(limits[0], limits[1], self.scale_bins),
            [limits[1]+range/self.scale_bins]
        ))
        self.zero_inverse = 0
        self.zero = 0
        self.ticks = self.default_ticks

    def logicle_ticks(self):
        top_tick = int(np.log10(self.xform.inverse(np.array([self.limits[1]])))[0]) + 1
        bottom_tick = int(np.log10(np.abs(self.xform.inverse(np.array([self.limits[0]]))))[0]) + 1
        threshold = 0.2
        cutoff_tick = np.argmax(np.diff(self.xform.apply(np.logspace(0, top_tick, top_tick+1))) > threshold)
        # print([self.logicle_w, self.logicle_m, top_tick, cutoff_tick])
        major_values = np.concatenate([
            -np.logspace(bottom_tick, cutoff_tick, max([bottom_tick-cutoff_tick+1, 0])),  # Negative values
            [0],
            np.logspace(cutoff_tick, top_tick, max([top_tick-cutoff_tick+1, 0]))  # Positive values
        ])
        minor_values = np.hstack([m * np.arange(0.1, 1, 0.1) if m != 0 else None for m in major_values])

        # Transform to plot coordinates
        trans_major_values = self.xform.apply(major_values)
        trans_minor_values = self.xform.apply(minor_values)

        superscripts = {
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"
        }

        def to_superscript(n):
            return "".join(superscripts.get(c, c) for c in str(n))

        # Create ticks with formatted labels
        major_ticks = []
        for val, raw in zip(trans_major_values, major_values):
            if raw == 0:
                label = "0"
            elif abs(raw) < 100:
                label = ""
            elif raw < 0:
                label = f"-10{to_superscript(int(np.log10(-raw)))}" if abs(raw) >= 10 else f"{raw:.1f}"
            else:
                label = f"10{to_superscript(int(np.log10(raw)))}" if raw >= 10 else f"{raw:.1f}"
            major_ticks.append((val, label))

        minor_ticks = [(val, '') for val in trans_minor_values]
        return [minor_ticks, major_ticks]

    def log_ticks(self):
        top_tick = int(np.log10(self.xform.inverse(np.array([self.limits[1]])))[0]) + 1
        major_values = np.logspace(0, top_tick, top_tick+1)
        minor_values = np.hstack([m * np.arange(0.1, 1, 0.1) if m != 0 else None for m in major_values])

        # Transform to plot coordinates
        trans_major_values = self.xform.apply(major_values)
        trans_minor_values = self.xform.apply(minor_values)

        superscripts = {
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"
        }

        def to_superscript(n):
            return "".join(superscripts.get(c, c) for c in str(n))

        # Create ticks with formatted labels
        major_ticks = []
        for val, raw in zip(trans_major_values, major_values):
            label = f"10{to_superscript(int(np.log10(raw)))}" if raw >= 10 else f"{raw:.1f}"
            major_ticks.append((val, label))
        minor_ticks = [(val, '') for val in trans_minor_values]
        return [minor_ticks, major_ticks]

    def linear_ticks(self):
        top_tick = 10**(int(np.log10(self.xform.inverse(self.limits[1]))) + 1)
        major_values = np.linspace(0,top_tick,10)
        minor_values = np.linspace(0,top_tick,50)

        # Transform to plot coordinates
        trans_major_values = self.xform.apply(major_values)
        trans_minor_values = self.xform.apply(minor_values)

        superscripts = {
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"
        }

        def to_superscript(n):
            return "".join(superscripts.get(c, c) for c in str(n))

        # Create ticks with formatted labels
        major_ticks = []
        for val, raw in zip(trans_major_values, major_values):
            if raw == 0:
                label = "0"
            elif abs(raw) < 100:
                label = ""
            elif raw < 0:
                label = f"-10{to_superscript(int(np.log10(-raw)))}" if abs(raw) >= 10 else f"{raw:.1f}"
            else:
                r,i = np.modf(np.log10(raw))
                i = int(i)
                r = 10**r
                label = f"{r:.0f}.10{to_superscript(i)}" if raw >= 10 else f"{raw:.1f}"
            major_ticks.append((val, label))

        minor_ticks = [(val, '') for val in trans_minor_values]
        return [minor_ticks, major_ticks]

    def default_ticks(self):
        return None

