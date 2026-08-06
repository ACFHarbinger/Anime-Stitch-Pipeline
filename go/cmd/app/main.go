// Command app is the example entry point for the go/ module template.
package main

import (
	"fmt"
	"os"

	"github.com/ACFHarbinger/dev-repo-template/go/internal/greet"
)

func main() {
	name := "world"
	if len(os.Args) > 1 {
		name = os.Args[1]
	}
	fmt.Println(greet.Greet(name))
}
