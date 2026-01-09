def to_latex_sci(x, decimals=2):
    s = f"{x:.{decimals}e}"
    mantissa, exp = s.split("e")
    exp = int(exp)  # removes leading zeros and '+' sign
    return rf"${mantissa} \times 10^{{{exp}}}$"