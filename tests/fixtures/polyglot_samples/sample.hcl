resource "aws_instance" "point" { ami = "abc" count = 1 }
variable "x" { default = 1 }
module "helper" { source = "./helper" }
locals { area = var.x * 2 }
output "area" { value = local.area }
