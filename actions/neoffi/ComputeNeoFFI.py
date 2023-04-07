class ComputeNeoFFI:
    def __init__(self, neuro_values, extra_values, off_values, ver_values, gew_values):
        self.neuro_values = neuro_values
        self.extra_values = extra_values
        self.off_values = off_values
        self.ver_values = ver_values
        self.gew_values = gew_values

    def Average(self,l):
        l = list(map(int, l))
        print(l)
        return sum(l) / len(l)

    # Scale factor according to the NEO-FFI manual.
    def Scale(self, i):
        i = i * 12
        return i

    def compute(self):
        neuro_mean = self.Average(self.neuro_values)
        extra_mean = self.Average(self.extra_values)
        off_mean = self.Average(self.off_values)
        ver_mean = self.Average(self.ver_values)
        gew_mean = self.Average(self.gew_values)

        # Scale mean values according to NEO-FFI manual
        # and convert into percentage, with
        # 60 being the max value
        max_value = 60
        neuro_skala = (self.Scale(neuro_mean) * 100) / max_value
        extra_skala = (self.Scale(extra_mean) * 100) / max_value
        off_skala = (self.Scale(off_mean) * 100) / max_value
        ver_skala = (self.Scale(ver_mean) * 100) / max_value
        gew_skala = (self.Scale(gew_mean) * 100) / max_value

        merged_results = [neuro_skala, extra_skala, off_skala, ver_skala, gew_skala]

        return merged_results




