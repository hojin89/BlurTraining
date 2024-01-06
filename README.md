<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Blur Training</h3>

  <p align="center">
    This GitHub repository contains the training code for data augmentation using blurry images, as detailed in the research paper "Improved modeling of human vision by incorporating robustness to blur in convolutional neural networks" authored by Hojin Jang and Frank Tong. For any inquiries, you can reach out to me at jangh@mit.edu.
    <br />
    <br />
    <a href="https://www.biorxiv.org/content/10.1101/2023.07.29.551089v1">Article</a>
    ·
    <a href="https://osf.io/upf5w/">Data Repository</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

This study hypothesized that the absence of blurred images in the training sets of neural networks might lead them to overly depend on high spatial frequency information for object recognition, resulting in divergences from human visual processing. This hypothesis was systematically assessed by comparing different training regimes involving both clear and blurred images (i.e., standard training, weak blur training, and strong blur training). The results demonstrated that networks trained with blurred images outperformed standard networks in predicting neural responses to objects under diverse viewing conditions. Additionally, these blur-trained networks developed an increased sensitivity to object shapes and increased robustness to various types of visual corruptions, aligning more closely with human perceptual processes. Our research underscores the importance of incorporating blur as a vital component in the training process for neural networks to develop representations of the visual world that are more congruent with human perception. Based on our results, we recommend the integration of blur as a standard image augmentation technique in the majority of computer vision tasks. 

### Codes

Below is the main code for blur training employed in this study. It contains a custom function for adjusting the sampling weights corresponding to a range of sigma values. Additionally, we integrated the Kornia library, speeding the process of Gaussian blurring via Tensor operations. Users can also refer to the latest PyTorch version, which includes the torchvision.transforms.GaussianBlur() function. This function allows for the blurring of images with a sigma value that is randomly determined. Tensorflow also offers the function tfa.image.gaussian_filter2d() to achieve similar outcomes. 

```
def add_blur_with(images, sigmas, weights):
    blurred_images = torch.zeros_like(images)
    normalize = transforms.Normalize(mean=[0.449], std=[0.226]) # grayscale

    for i in range(images.size(0)): # Batch size
        image = images[i, :, :, :]
        weights = numpy.asarray(weights).astype('float64')
        weights = weights / numpy.sum(weights)
        sigma = choice(sigmas, 1, p=weights)[0]
        kernel_size = 2 * math.ceil(2.0 * sigma) + 1

        if sigma == 0:
            blurred_image = image
        else:
            blurred_image = kornia.gaussian_blur2d(torch.unsqueeze(image, dim=0), kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[0, :, :, :]
        blurred_image = normalize(blurred_image)
        blurred_images[i] = blurred_image

    blurred_images = blurred_images.repeat(1, 3, 1, 1) # Grayscale to RGB
    return blurred_images

```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This research was supported by the following grants from the National Eye Institute, National Institutes of Health (NEI/NIH): R01EY035157 and R01EY029278 to [Frank Tong](http://www.psy.vanderbilt.edu/tonglab/web/Home.html), and P30EY008126 to the Vanderbilt Vision Research Center.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
