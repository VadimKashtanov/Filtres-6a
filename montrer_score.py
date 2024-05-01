import struct as st

import matplotlib.pyplot as plt

with open("mdl", "rb") as co:
	bins = co.read()
	(I,) = st.unpack('I', bins[:4])
	s = st.unpack('f'*I, bins[4:])

with open("mdl_validation", "rb") as co:
	bins = co.read()
	(I,) = st.unpack('I', bins[:4])
	s_valid = st.unpack('f'*I, bins[4:])

i = int(len(s_valid) / len(s))

plt.plot(s_valid, label='validation')
plt.plot([_s for _s in s for _ in range(i)], 'x-', label='entrainnement')
plt.legend()
plt.show()