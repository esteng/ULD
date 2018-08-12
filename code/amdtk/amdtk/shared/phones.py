
class TIMIT_phones:

	# List of lists of phone symbols in TIMIT.
	# The first symbol in each list is the one used in the textgrids
	# (they have been preprocessed to substitute the latter symbols for the first symbol).
	# The other symbols are included either for reference, or because they appear in
	# the pronunciation dictionary and are therefore needed to construct the top-level phone strings.

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
		['l', 'el'],
		['m', 'em'],
		['n', 'en'],
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

	# Make sure the two lookup dictionaries are consistent
	for integer, phone in int_to_phone.items():
		assert(phone_to_int[phone]==integer)
