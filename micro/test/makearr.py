import sys
from PIL import Image

if len(sys.argv) != 3:
    print('Usage: python {} image.gif array.c'.format(sys.argv[0]))
    sys.exit(1)

im = Image.open(sys.argv[1])

count = 1

with open(sys.argv[2], 'w') as f:
    f.write('{\n')
    while True:
        rgb_im = im.convert('RGBA')
        size = rgb_im.size
        print(size)

        array = []
        for y in range(size[0]):
            for x in range(size[1]):
                # rgb = rgb_im.getpixel((x, y))

                # reverse white and black
                # if rgb == (255, 255, 255):
                #     rgb = (0, 0, 0)
                # elif rgb == (0, 0, 0):
                #     rgb = (255, 255, 255)
                
                # for c in rgb:
                #     array.append(c)

                rgba = rgb_im.getpixel((x, y))

                if rgba[3] == 0:
                    array.append(0)
                    array.append(0)
                    array.append(0)
                else:
                    array.append(rgba[0])
                    array.append(rgba[1])
                    array.append(rgba[2])


        # print(len(array))

        f.write('{')
        f.write(str(array)[1:-1])
        f.write('},\n')

        # To iterate through the entire gif
        try:
            im.seek(im.tell()+1)

            count += 1
        except EOFError:
            break # end of sequence

    f.write('};\n')

print(count)