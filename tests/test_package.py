import optimap as om


def test_version_present():
    assert om.__version__

def test_verbose():
    om.set_verbose(True)
    assert om.is_verbose() is True
    om.set_verbose(False)
    assert om.is_verbose() is False
