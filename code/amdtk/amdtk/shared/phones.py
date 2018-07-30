
class TIMIT_phones:

	__phone_symbols__ = [
		['','sil','sp'],
		['aa'],
		['ae'],
		['ah','ax','ix'],
		['ao'],
		['aw'],
		['ay'],
		['b'],
		['ch'],
		['d'],
		['dh'],
		['dx'],
		['eh'],
		['er', 'axr'],
		['ey'],
		['f'],
		['g'],
		['hh'],
		['ih'],
		['iy'],
		['jh'],
		['k'],
		['l','el'],
		['m'],
		['n'],
		['ng'],
		['ow'],
		['oy'],
		['p'],
		['q'],
		['r'],
		['s'],
		['sh'],
		['t'],
		['th'],
		['uh'],
		['uw'],
		['v'],
		['w'],
		['y'],
		['z'],
		['zh']
	]

	phone_to_int = {}
	int_to_phone = {}

	for i, phone_set in enumerate(__phone_symbols__):
		int_to_phone[i] = phone_set[0]
		for phone in phone_set:
			phone_to_int[phone] = i

	for integer, phone in int_to_phone.items():
		assert(phone_to_int[phone]==integer)
