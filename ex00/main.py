from pattern import Checker, Circle, Spectrum

if __name__ == '__main__':
    c = Checker(250, 25)
    c.draw()
    c.show()

    circle = Circle(200, 20, (50, 50))
    circle.draw()
    circle.show()

    s = Spectrum(200)
    s.draw()
    s.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
